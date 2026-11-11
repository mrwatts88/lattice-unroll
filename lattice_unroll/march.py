"""lattice-march: carry the scroll axis along the scroll, starting from one point placed by eye.

    lattice-march STORE_DIR AXIS.json Z0 Z1 WORK_DIR --start X Y Z [--step 200] [lattice-unroll options]

lattice-unroll needs the scroll axis (the line the sheet winds round) at every height it solves. Place one
point by eye at height Z, inside the innermost loop of a cross-section (lattice-peek draws one). This
solves a thin band of --step slices round Z with that point, then thin bands one after the other up to Z1
and down to Z0, each with the axis known so far, and after every band adds to AXIS.json where the
innermost complete winding of that band sits (its middle, every 80 slices). No surfaces are written.

The bands are solved one by one, so their winding numbers are not tied to each other: this is only for
the axis. Afterwards run lattice-unroll once on the whole range with AXIS.json. The winding sense is read
from the first band and passed on to the rest (thin bands cannot always read it themselves). Without
--start, AXIS.json must exist and its first control point is the start.
"""
import argparse
import contextlib
import json
import os
import sys

import numpy as np

from . import lattice


def merge(axis_path, band_axis_path):
    """add the band's control points to the axis file (one per z, newest wins); points placed by eye go once
    there are measured ones"""
    a = json.load(open(axis_path))
    b = json.load(open(band_axis_path))
    pts = {int(c["z"]): c for c in a["control_points"] if c.get("winding") is not None}
    by_eye = [c for c in a["control_points"] if c.get("winding") is None]
    for c in b["control_points"]:
        pts[int(c["z"])] = c
    cps = sorted(pts.values(), key=lambda c: c["z"]) or by_eye
    a["control_points"] = cps
    a["source"] = "lattice-march: middle of the innermost complete winding, band by band"
    json.dump(a, open(axis_path, "w"), indent=1)
    return cps


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("store")
    ap.add_argument("axis")
    ap.add_argument("z0", type=int)
    ap.add_argument("z1", type=int)
    ap.add_argument("work_dir")
    ap.add_argument("--start", type=float, nargs=3, metavar=("X", "Y", "Z"), help="the point placed by eye")
    ap.add_argument("--step", type=int, default=200, help="slices per band")
    args, extra = ap.parse_known_args(argv)
    os.makedirs(args.work_dir, exist_ok=True)
    if args.start:
        x, y, z = args.start
        json.dump({"control_points": [{"x": x, "y": y, "z": int(z)}], "source": "placed by eye"}, open(args.axis, "w"), indent=1)
    if not os.path.exists(args.axis):
        raise SystemExit(f"{args.axis} does not exist: give --start X Y Z")
    zs = int(json.load(open(args.axis))["control_points"][0]["z"])
    if not args.z0 <= zs < args.z1:
        raise SystemExit(f"the start point (z {zs}) must lie inside z {args.z0} to {args.z1}")
    h = args.step // 2
    bands = [(max(args.z0, zs - h), min(args.z1, zs + h))]
    a = bands[0][1]
    while a < args.z1:
        bands.append((a, min(a + args.step, args.z1)))
        a += args.step
    b = bands[0][0]
    while b > args.z0:
        bands.append((max(b - args.step, args.z0), b))
        b -= args.step
    sense = None
    if "--sense" in extra:
        sense = int(extra[extra.index("--sense") + 1])
    print(f"{len(bands)} bands of up to {args.step} slices, starting at z {zs}", flush=True)
    for a, b in bands:
        wd = os.path.join(args.work_dir, f"band_{a}_{b}")
        log = wd + ".log"
        cmd = [args.store, args.axis, str(a), str(b), wd, "--no-surfaces", "--tail-rel", "0.001"] + extra
        if sense is not None and "--sense" not in extra:
            cmd += ["--sense", str(sense)]
        with open(log, "w") as fh, contextlib.redirect_stdout(fh):
            try:
                lattice.main(cmd)
                failed = None
            except SystemExit as e:
                failed = str(e)
        if failed:
            print(f"band z {a} to {b} failed: {failed}\n(log: {log})", file=sys.stderr)
            raise SystemExit(1)
        if sense is None:
            sense = int(np.load(os.path.join(wd, "wind.npz"))["sense"])
            print(f"winding sense read from the first band: {sense:+d}", flush=True)
        cps = merge(args.axis, os.path.join(wd, "axis_inner.json"))
        inner = json.load(open(os.path.join(wd, "axis_inner.json")))["control_points"]
        got = "; ".join(f"z {c['z']}: x {c['x']:.0f}, y {c['y']:.0f}" for c in inner) or "no complete winding found"
        print(f"band z {a} to {b}: {got}. Axis file: {len(cps)} points, z {cps[0]['z']} to {cps[-1]['z']}", flush=True)
        for f in os.listdir(wd):
            if f.startswith("band_") and f.endswith(".npz"):
                os.remove(os.path.join(wd, f))
    print(f"done. Next: lattice-unroll {args.store} {args.axis} {args.z0} {args.z1} OUT_DIR --sense {sense} [options]")


if __name__ == "__main__":
    main()
