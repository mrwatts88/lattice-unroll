"""trace_peek.py: look at one cross-section of a scroll's public trace store without downloading the store.

    uv run --with numpy --with pillow python cinder/trace_peek.py SCROLL OUT.png [--z-frac 0.5] [--down 6] [--dir runs]

Finds the scroll's volume folder under https://dl.ash2txt.org/datasets/spiral_datasets/SCROLL/, downloads
the three small index files (offsets, z bounds, families: about 250 MB) to DIR/tracks_SCROLL/ if they are
not there, then reads only the h traces of one slice from coordinates.i32 with range requests and draws
them (grey points, a coordinate grid every 500 voxels). For choosing which scroll to map next and for
placing a first axis point by eye: a scroll whose turns can be followed round a hollow middle suits
lattice.py, a scroll crushed into zig-zag folds does not.
"""
import argparse
import json
import os
import re
import urllib.request

import numpy as np
from PIL import Image, ImageDraw

BASE = "https://dl.ash2txt.org/datasets/spiral_datasets"


def get(url, rng=None):
    req = urllib.request.Request(url, headers={"Range": f"bytes={rng[0]}-{rng[1]}"} if rng else {})
    return urllib.request.urlopen(req, timeout=300).read()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scroll")
    ap.add_argument("out")
    ap.add_argument("--z-frac", type=float, default=0.5)
    ap.add_argument("--z", type=int)
    ap.add_argument("--down", type=int, default=6)
    ap.add_argument("--dir", default="runs")
    args = ap.parse_args()
    vols = re.findall(r'href="(\d+)/"', get(f"{BASE}/{args.scroll}/").decode())
    vol = vols[-1]
    stores = re.findall(r'href="([^"]+\.vctracks)/"', get(f"{BASE}/{args.scroll}/{vol}/tracks/").decode())
    url = f"{BASE}/{args.scroll}/{vol}/tracks/{stores[0]}"
    d = os.path.join(args.dir, f"tracks_{args.scroll}")
    os.makedirs(d, exist_ok=True)
    for f in ("metadata.json", "family_codes.i8", "offsets.i64", "z_bounds.i32"):
        if not os.path.exists(os.path.join(d, f)):
            open(os.path.join(d, f), "wb").write(get(f"{url}/{f}"))
    meta = json.load(open(os.path.join(d, "metadata.json")))
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
