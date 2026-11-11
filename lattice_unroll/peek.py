"""lattice-peek: one cross-section of a scroll's public trace store, without downloading the whole store.

    lattice-peek SCROLL OUT.png [--z 10237 | --z-frac 0.5] [--down 6] [--stores stores]

Fetches the scroll's index files (about 300 MB) to STORES/SCROLL/ if they are not there (the same place
lattice-fetch puts them), then reads only the h traces of one slice from coordinates.i32 with range
requests and draws them: grey points, a coordinate grid every 500 voxels. For choosing a scroll and for
placing the first axis point by eye: a scroll whose turns can be followed round a hollow middle suits
lattice-unroll, a scroll crushed into zig-zag folds does not. The axis point goes inside the innermost
loop. --down 2 gives a sharper picture of the middle.
"""
import argparse
import json
import os

import numpy as np
from PIL import Image, ImageDraw

from .stores import fetch, get


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scroll")
    ap.add_argument("out")
    ap.add_argument("--z-frac", type=float, default=0.5)
    ap.add_argument("--z", type=int)
    ap.add_argument("--down", type=int, default=6)
    ap.add_argument("--stores", default="stores")
    args = ap.parse_args()
    url, d, meta = fetch(args.scroll, args.stores, index_only=True)
    vol = url.split("/")[-3]
    off = np.fromfile(os.path.join(d, "offsets.i64"), dtype="<i8")
    zb = np.fromfile(os.path.join(d, "z_bounds.i32"), dtype="<i4").reshape(-1, 2)
    fam = np.fromfile(os.path.join(d, "family_codes.i8"), dtype=np.int8)
    ln = off[1:] - off[:-1]
    zs = np.unique(zb[(fam == 0) & (ln >= 6), 0])
    z = int(zs[np.argmin(np.abs(zs - args.z))]) if args.z is not None else int(zs[int(args.z_frac * (len(zs) - 1))])
    tr = np.flatnonzero((fam == 0) & (zb[:, 0] == z) & (ln >= 6))
    # the traces of one slice lie in a few runs of the file: fetch each run once
    brk = np.flatnonzero(np.diff(tr) > 200)
    runs = zip(np.r_[tr[0], tr[brk + 1]], np.r_[tr[brk], tr[-1]])
    pts = []
    keep = np.zeros(len(fam), bool)
    keep[tr] = True
    for a, b in runs:
        buf = np.frombuffer(get(f"{url}/coordinates.i32", (int(off[a]) * 12, int(off[b + 1]) * 12 - 1)), dtype="<i4").reshape(-1, 3)
        m = np.repeat(keep[a:b + 1], ln[a:b + 1])
        pts.append(buf[m])
    pts = np.concatenate(pts)
    x, y = pts[:, 2], pts[:, 1]
    down = args.down
    X0, Y0 = int(x.min()) - 50, int(y.min()) - 50
    W, H = (int(x.max()) + 50 - X0) // down + 1, (int(y.max()) + 50 - Y0) // down + 1
    img = np.zeros((H, W, 3), np.uint8)
    img[(y - Y0) // down, (x - X0) // down] = (200, 200, 200)
    im = Image.fromarray(img)
    dr = ImageDraw.Draw(im)
    for gx in range((X0 // 500 + 1) * 500, X0 + W * down, 500):
        dr.line([((gx - X0) // down, 0), ((gx - X0) // down, H)], fill=(60, 60, 140))
        dr.text(((gx - X0) // down + 2, 2), str(gx), fill=(120, 160, 255))
    for gy in range((Y0 // 500 + 1) * 500, Y0 + H * down, 500):
        dr.line([(0, (gy - Y0) // down), (W, (gy - Y0) // down)], fill=(60, 60, 140))
        dr.text((2, (gy - Y0) // down + 2), str(gy), fill=(120, 160, 255))
    im.save(args.out)
    print(f"{args.scroll} volume {vol}: {meta['track_count'] / 1e6:.1f} M traces, {meta['point_count'] / 1e6:.0f} M points, h slices z {int(zs.min())} to {int(zs.max())}; "
          f"slice z {z}: {len(tr)} traces, {len(pts)} points -> {args.out} {im.size}")


if __name__ == "__main__":
    main()
