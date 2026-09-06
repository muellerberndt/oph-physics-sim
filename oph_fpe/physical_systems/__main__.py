"""Read-only physical-model queries; no external service or credentials."""
import argparse
import json
from pathlib import Path

import numpy as np

from . import WhitneySystem
from .free_scalar import FreeScalar
from .bridges import ControlledBridges


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
    sub.add_parser("bridges", help="controlled quantum preparation and magnetic continuum evidence")
    packet = sub.add_parser("packet", help="full56D neutral preparation, not quantum propagation")
    packet.add_argument("--sample", type=int, default=0)
    packet.add_argument("--width", choices=("1/4", "1/2", "1"), default="1/2")
    overlap = sub.add_parser("packet-overlap")
    overlap.add_argument("angle", type=float)
    overlap.add_argument("--sample", type=int, default=0)
    overlap.add_argument("--width", choices=("1/4", "1/2", "1"), default="1/2")
    magnetic = sub.add_parser("magnetic-continuum")
    magnetic.add_argument("--refinement", type=int)
    magnetic.add_argument("--steps", type=int)
    sub.add_parser("magnetic-field").add_argument("position", nargs=3, type=float)
    sub.add_parser("charged-enclosure").add_argument("time", help="exact rational model time in[0,2], e.g.1/80")
    args = parser.parse_args(argv)
    try:
        if args.query in ("bridges", "packet", "packet-overlap", "magnetic-continuum", "magnetic-field", "charged-enclosure"):
            bridge = ControlledBridges.load(args.rer_root)
            if args.query == "bridges": result = bridge.describe()
            elif args.query == "packet": result = bridge.packet(args.sample, args.width)
            elif args.query == "packet-overlap": result = bridge.packet_overlap(args.angle, args.sample, args.width)
            elif args.query == "magnetic-continuum": result = bridge.magnetic_continuum(args.refinement, steps=args.steps)
            elif args.query == "magnetic-field": result = bridge.magnetic_field(args.position)
            else: result = bridge.charged_enclosure(args.time)
        else:
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
