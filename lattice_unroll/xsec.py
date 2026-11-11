"""lattice-xsec: one cross-section of a scroll with surfaces drawn on it, to see whether they follow the sheets.

    lattice-xsec STORE_DIR Z OUT.png --surfaces DIR FMT [--surfaces DIR2 FMT2 ...] [--crop X0 Y0 X1 Y1] [--down 2]

The background is the h traces of slice Z from the trace store (grey: every predicted sheet). Each
surface (tifxyz, one per winding) is cut at height Z and drawn in its own colour, neighbouring windings in
clearly different colours. A good surface stays on one grey sheet all the way round. Two surfaces on one
sheet show as two colours on top of each other, and a jump to the next sheet as a colour that crosses the
gap. Several --surfaces sets come out side by side, with the same crop.
"""
import argparse
import colorsys
import json
import os
import re

import numpy as np
import tifffile
from PIL import Image, ImageDraw


def slice_traces(store, z):
    meta = json.load(open(os.path.join(store, "metadata.json")))
    off = np.fromfile(os.path.join(store, "offsets.i64"), dtype="<i8")
    zb = np.fromfile(os.path.join(store, "z_bounds.i32"), dtype="<i4").reshape(-1, 2)
    fam = np.fromfile(os.path.join(store, "family_codes.i8"), dtype=np.int8)
    coords = np.memmap(os.path.join(store, "coordinates.i32"), dtype="<i4", mode="r", shape=(int(meta["point_count"]), 3))
    hz = np.unique(zb[fam == 0, 0])
    z = int(hz[np.argmin(np.abs(hz - z))])
    tr = np.flatnonzero((fam == 0) & (zb[:, 0] == z) & (off[1:] - off[:-1] >= 6))
    brk = np.flatnonzero(np.diff(tr) > 200)
    pts = []
    for a, b in zip(np.r_[tr[0], tr[brk + 1]], np.r_[tr[brk], tr[-1]]):
        buf = np.asarray(coords[int(off[a]):int(off[b + 1])])
        keep = np.zeros(b - a + 1, bool)
        keep[tr[(tr >= a) & (tr <= b)] - a] = True
        pts.append(buf[np.repeat(keep, off[a + 1:b + 2] - off[a:b + 1])])
    return z, np.concatenate(pts)


def cut(d, z):
    """where a tifxyz surface crosses height z: polylines of (x, y), one per run of columns"""
    P = np.stack([tifffile.imread(os.path.join(d, f"{a}.tif")).astype(np.float64) for a in "xyz"], -1)
    ok = (P[..., 0] > 0) & (P[..., 2] > 0)
    zz = np.where(ok, P[..., 2], np.nan)
    lo, hi = zz[:-1], zz[1:]
    hit = ok[:-1] & ok[1:] & ((lo - z) * (hi - z) <= 0) & (lo != hi)
    f = np.where(hit, (z - lo) / np.where(lo != hi, hi - lo, 1), 0)
    xy = P[:-1, :, :2] * (1 - f[..., None]) + P[1:, :, :2] * f[..., None]
    W = P.shape[1]
    lines, cur = [], []
    for c in range(W):
        rows = np.flatnonzero(hit[:, c])
        if len(rows):
            cur.append(xy[rows[0], c])
        elif cur:
            lines.append(np.array(cur)); cur = []
    if cur:
        lines.append(np.array(cur))
    return lines


def colour(k):
    r, g, b = colorsys.hsv_to_rgb((k * 0.381966) % 1.0, 0.85, 1.0)
    return int(r * 255), int(g * 255), int(b * 255)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("store")
    ap.add_argument("z", type=int)
    ap.add_argument("out")
    ap.add_argument("--surfaces", nargs=2, action="append", metavar=("DIR", "FMT"), required=True)
    ap.add_argument("--labels", nargs="+", help="one title per --surfaces set")
    ap.add_argument("--crop", type=int, nargs=4, metavar=("X0", "Y0", "X1", "Y1"))
    ap.add_argument("--down", type=float, default=2.0)
    ap.add_argument("--width", type=int, default=2)
    args = ap.parse_args()
    z, pts = slice_traces(args.store, args.z)
    x, y = pts[:, 2], pts[:, 1]
    X0, Y0, X1, Y1 = args.crop if args.crop else (int(x.min()) - 40, int(y.min()) - 40, int(x.max()) + 40, int(y.max()) + 40)
    s = 1.0 / args.down
    W, H = int((X1 - X0) * s), int((Y1 - Y0) * s)
    panels = []
    for i, (d, fmt) in enumerate(args.surfaces):
        img = np.full((H, W, 3), 12, np.uint8)
        m = (x >= X0) & (x < X1) & (y >= Y0) & (y < Y1)
        img[((y[m] - Y0) * s).astype(int).clip(0, H - 1), ((x[m] - X0) * s).astype(int).clip(0, W - 1)] = (110, 110, 110)
        im = Image.fromarray(img)
        dr = ImageDraw.Draw(im)
        pat = re.compile("^" + re.escape(fmt).replace(r"\{:03d\}", r"(\d+)").replace(r"\{\}", r"(\d+)") + "$")
        n = 0
        for name in sorted(os.listdir(d)):
            mm = pat.match(name)
            if not mm or not os.path.exists(os.path.join(d, name, "x.tif")):
                continue
            k = int(mm.group(1))
            for ln in cut(os.path.join(d, name), z):
                if len(ln) >= 2:
                    dr.line([((px - X0) * s, (py - Y0) * s) for px, py in ln], fill=colour(k), width=args.width)
            n += 1
        title = (args.labels[i] if args.labels and i < len(args.labels) else os.path.basename(d.rstrip("/"))) + f"  (z {z}, {n} surfaces)"
        dr.rectangle([0, 0, 9 * len(title) + 12, 22], fill=(0, 0, 0))
        dr.text((6, 5), title, fill=(255, 255, 255))
        panels.append(im)
    gap = 8
    out = Image.new("RGB", (W * len(panels) + gap * (len(panels) - 1), H), (255, 255, 255))
    for i, p in enumerate(panels):
        out.paste(p, (i * (W + gap), 0))
    out.save(args.out)
    print(f"slice z {z}: {len(pts)} trace points, crop x {X0} to {X1}, y {Y0} to {Y1} -> {args.out} {out.size}")


if __name__ == "__main__":
    main()
