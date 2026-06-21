#!/usr/bin/env bash
set -euo pipefail
ROOT="${OPH_FPE_ROOT:-$(pwd)}"
PACK_DIR="${OPH_DISTRIBUTED_PACK_DIR:-$ROOT/distributed/oph_distributed_federated_4m64shards_20260621}"
RUN_ROOT="${OPH_DISTRIBUTED_RUN_ROOT:-$ROOT/runs/oph_distributed_federated_4m64shards_20260621}"
python3 -m oph_fpe.cli reduce-distributed-oph-universe \
  --manifest "$PACK_DIR/distributed_universe_manifest.json" \
  --shard-root "$RUN_ROOT/shards" \
  --out-dir "$RUN_ROOT/reduced"
