"""Read-only physical-model queries; no external service or credentials."""
import argparse
import json
from pathlib import Path

import numpy as np

from . import WhitneySystem
from .free_scalar import FreeScalar


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rer-root", type=Path)
    sub = parser.add_subparsers(dest="query", required=True)
    for name in ("describe", "mesh", "real_continuum", "quantum"):
        sub.add_parser(name)
    observer = sub.add_parser("observer")
    observer.add_argument("patch", type=int); observer.add_argument("frame", type=int)
    event = sub.add_parser("event"); event.add_argument("event_id", type=int)
    field = sub.add_parser("field")
    field.add_argument("frame", type=int); field.add_argument("cell", type=int)
    field.add_argument("barycentric", type=float, nargs=4)
    for name in ("energy", "clock"):
        sub.add_parser(name).add_argument("frame", type=int)
    free = sub.add_parser("free-state", help="new all-mode free-field coherent calculation on the same mesh")
    free.add_argument("--time", type=float, default=1.)
    args = parser.parse_args(argv)
    try:
        system = WhitneySystem.load(args.rer_root)
        if args.query == "observer": result = system.observer(args.patch, args.frame)
        elif args.query == "event": result = system.event(args.event_id)
        elif args.query == "field": result = system.field(args.frame, args.cell, args.barycentric)
        elif args.query in ("energy", "clock"): result = getattr(system, args.query)(args.frame)
        elif args.query == "free-state":
            model = FreeScalar(system.mesh())
            initial = np.exp(-np.sum(model.vertices**2, axis=1)/2)
            state = model.prepare(initial, np.zeros(len(initial)))
            result = state.state(args.time)
            result["preparation"] = {"field": initial.tolist(), "velocity": [0.]*len(initial),
                                     "scope": "declared Gaussian-shaped nodal mean; vacuum covariance"}
            result["geometry_provenance"] = system.mesh()["provenance"]
        else: result = getattr(system, args.query)()
        print(json.dumps(result, indent=2, allow_nan=False))
    except (ValueError, OSError, KeyError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
