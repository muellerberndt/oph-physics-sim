#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 3 ]; then
  echo "usage: $0 WORKER_INDEX WORKER_COUNT PARALLELISM" >&2
  exit 2
fi

WORKER_INDEX="$1"
WORKER_COUNT="$2"
PARALLELISM="$3"
ROOT="${OPH_FPE_ROOT:-$(pwd)}"
PACK_DIR="${OPH_DISTRIBUTED_PACK_DIR:-$ROOT/distributed/oph_distributed_federated_4m64shards_20260621}"
RUN_ROOT="${OPH_DISTRIBUTED_RUN_ROOT:-$ROOT/runs/oph_distributed_federated_4m64shards_20260621}"
LOG_DIR="$RUN_ROOT/logs"
PYTHON_BIN="${OPH_PYTHON:-python3}"
mkdir -p "$RUN_ROOT/shards" "$LOG_DIR"

python3 - "$PACK_DIR/distributed_universe_manifest.json" "$PACK_DIR" "$WORKER_INDEX" "$WORKER_COUNT" > "$RUN_ROOT/worker_${WORKER_INDEX}_configs.txt" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
pack_dir = Path(sys.argv[2])
worker_index = int(sys.argv[3])
worker_count = int(sys.argv[4])
for shard in manifest["shards"]:
    if int(shard["worker_index"]) == worker_index % worker_count:
        print(pack_dir / shard["config_path"])
PY

run_one() {
  set -euo pipefail
  cfg="$1"
  shard="$(basename "$cfg" .yml)"
  echo "[$(date -u +%FT%TZ)] start $shard" | tee -a "$LOG_DIR/worker_${WORKER_INDEX}.log"
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
    "$PYTHON_BIN" -m oph_fpe.cli run-oph-universe \
      --config "$cfg" \
      --out-dir "$RUN_ROOT/shards" \
      --max-screen-points 8000 \
      --max-observers 4096 \
      --max-h3-objects 1024 \
      > "$LOG_DIR/${shard}.stdout.json" \
      2> "$LOG_DIR/${shard}.stderr.log"
  echo "[$(date -u +%FT%TZ)] done $shard" | tee -a "$LOG_DIR/worker_${WORKER_INDEX}.log"
}
export -f run_one
export LOG_DIR RUN_ROOT WORKER_INDEX PYTHON_BIN

if [ ! -s "$RUN_ROOT/worker_${WORKER_INDEX}_configs.txt" ]; then
  echo "No shard configs assigned to worker $WORKER_INDEX" | tee -a "$LOG_DIR/worker_${WORKER_INDEX}.log"
  exit 0
fi

xargs -n 1 -P "$PARALLELISM" -I {} bash -lc 'run_one "$@"' _ {} < "$RUN_ROOT/worker_${WORKER_INDEX}_configs.txt"
