"""lattice-centre: find a point inside the innermost loop of a scroll at one height, without looking.

    lattice-centre STORE_DIR Z0 Z1 [--slices 5] [--png OUT.png]

Prints X Y Z as its last line, ready for `lattice-march --start X Y Z`. Several slices inside Z0 to Z1 are
tried and the one where the point has the most clear space round it is kept: a hollow core is the easiest
place to be sure of. Crushed slices without a core lose to it.

Every h trace of the slice is a piece of a sheet. Seen from a point c, a piece sweeps through some angle.
A whole turn of sheet that goes round c sweeps 2 pi whatever c's position inside it, and a turn that does
not go round c sweeps much less (it goes out and comes back). Adding the swept angle of every piece (each
taken as a size, since pieces are traced in either direction) gives about 2 pi times the number of turns
round c. That count is largest, and flat, inside the innermost loop, and falls by one at every sheet
crossed going outward. So the point is where the count is largest: found on a coarse grid over the whole
slice, then on a fine grid round the best coarse point, and taken as the middle of the cells within one
turn of the best. The value in turns is printed too: it should be close to the number of windings.
"""
import argparse
import json
import os

import numpy as np
from scipy.spatial import cKDTree


def slice_pieces(store, z, step=12.0):
    """the h traces of the slice nearest z, each as a polyline thinned to about `step` voxels between points.
    Returns the slice height, the points (n, 2) as x, y and the index where each piece starts."""
    meta = json.load(open(os.path.join(store, "metadata.json")))
    off = np.fromfile(os.path.join(store, "offsets.i64"), dtype="<i8")
    zb = np.fromfile(os.path.join(store, "z_bounds.i32"), dtype="<i4").reshape(-1, 2)
    fam = np.fromfile(os.path.join(store, "family_codes.i8"), dtype=np.int8)
    coords = np.memmap(os.path.join(store, "coordinates.i32"), dtype="<i4", mode="r", shape=(int(meta["point_count"]), 3))
    hz = np.unique(zb[fam == 0, 0])
    z = int(hz[np.argmin(np.abs(hz - z))])
    tr = np.flatnonzero((fam == 0) & (zb[:, 0] == z) & (off[1:] - off[:-1] >= 6))
    if len(tr) == 0:
        raise SystemExit(f"no h traces at z {z}")
    # h traces of one slice sit together in the file: read them in a few contiguous runs
    brk = np.flatnonzero(np.diff(tr) > 200)
    pts, starts, n = [], [], 0
    for a, b in zip(np.r_[tr[0], tr[brk + 1]], np.r_[tr[brk], tr[-1]]):
        buf = np.asarray(coords[int(off[a]):int(off[b + 1])])
        for t in tr[(tr >= a) & (tr <= b)]:
            p = buf[int(off[t] - off[a]):int(off[t + 1] - off[a]), :][:, [2, 1]].astype(np.float64)   # x, y
            d = np.r_[0.0, np.cumsum(np.hypot(*np.diff(p, axis=0).T))]
            keep = np.unique(np.r_[np.searchsorted(d, np.arange(0.0, d[-1], step)), len(p) - 1])
            if len(keep) < 2:
                continue
            starts.append(n)
            pts.append(p[keep])
            n += len(keep)
    return z, np.concatenate(pts), np.array(starts)


def turns(pts, starts, cx, cy, chunk=128):
    """for each candidate point (cx[i], cy[i]): the sum over pieces of the size of the angle each piece sweeps,
    in turns"""
    seg_ok = np.ones(len(pts) - 1, bool)
    seg_ok[starts[1:] - 1] = False                       # no segment from the end of one piece to the next
    a, b = pts[:-1][seg_ok], pts[1:][seg_ok]
    piece = np.cumsum(np.isin(np.arange(len(pts) - 1), starts))[seg_ok] - 1
    first = np.r_[0, np.flatnonzero(np.diff(piece)) + 1]
    out = np.empty(len(cx))
    for i in range(0, len(cx), chunk):
        c = np.stack([cx[i:i + chunk], cy[i:i + chunk]], -1)[:, None, :]
        u, v = a[None] - c, b[None] - c
        cr = u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]
        dt = u[..., 0] * v[..., 0] + u[..., 1] * v[..., 1]
        ang = np.arctan2(cr, dt)
        out[i:i + chunk] = np.abs(np.add.reduceat(ang, first, axis=1)).sum(1) / (2 * np.pi)
    return out


def find(store, z, coarse=48.0, fine=8.0, band=3.0):
    z, pts, starts = slice_pieces(store, z)
    x0, y0 = pts.min(0)
    x1, y1 = pts.max(0)
    gx, gy = np.meshgrid(np.arange(x0, x1, coarse), np.arange(y0, y1, coarse))
    T = turns(pts, starts, gx.ravel(), gy.ravel()).reshape(gx.shape)
    j = np.argmax(T)
    bx, by = gx.ravel()[j], gy.ravel()[j]
    r = 8 * coarse
    fx, fy = np.meshgrid(np.arange(bx - r, bx + r + fine, fine), np.arange(by - r, by + r + fine, fine))
    fx, fy = fx.ravel(), fy.ravel()
    F = turns(pts, starts, fx, fy)
    # Traces have gaps, so the count is a broad hill, not a flat top. Near its top, the inside of the innermost
    # loop is the widest place with no sheet in it: take the cell farthest from every trace point.
    top = F >= F.max() - band
    clear = cKDTree(pts).query(np.stack([fx[top], fy[top]], -1))[0]
    k = np.argmax(clear)
    px, py = float(fx[top][k]), float(fy[top][k])
    at = float(turns(pts, starts, np.array([px]), np.array([py]))[0])
    return dict(z=z, x=px, y=py, turns=at, best=float(F.max()), clear=float(clear[k]), coarse=(gx, gy, T), pts=pts)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("store")
    ap.add_argument("z0", type=int)
    ap.add_argument("z1", type=int)
    ap.add_argument("--slices", type=int, default=5, help="slices tried, evenly spaced inside Z0 to Z1")
    ap.add_argument("--png", help="draw the chosen slice's traces with the turn count and the point found")
    args = ap.parse_args(argv)
    zs = np.linspace(args.z0, args.z1, args.slices + 2)[1:-1].round().astype(int) if args.slices > 1 else [(args.z0 + args.z1) // 2]
    best = None
    for z in zs:
        r = find(args.store, int(z))
        print(f"slice z {r['z']}: point x {r['x']:.0f}, y {r['y']:.0f}; {r['turns']:.1f} turns round it "
              f"(best {r['best']:.1f}); nearest trace {r['clear']:.0f} voxels away", flush=True)
        if best is None or r["clear"] > best["clear"]:
            best = r
    r = best
    print(f"chosen: slice z {r['z']}, the widest clear space round the point (a hollow core is easiest to be inside)")
    if args.png:
        from PIL import Image, ImageDraw
        gx, gy, T = r["coarse"]
        step = gx[0, 1] - gx[0, 0]
        x0, y0 = gx[0, 0], gy[0, 0]
        s = 0.25
        W, H = int(gx.shape[1] * step * s) + 1, int(gx.shape[0] * step * s) + 1
        heat = (255 * T / max(T.max(), 1e-9)).astype(np.uint8)
        img = np.array(Image.fromarray(heat).resize((W, H), Image.NEAREST))
        rgb = np.stack([img // 2, img // 3, img // 2], -1).astype(np.uint8)
        p = r["pts"]
        px = ((p[:, 0] - x0) * s).astype(int).clip(0, W - 1)
        py = ((p[:, 1] - y0) * s).astype(int).clip(0, H - 1)
        rgb[py, px] = (200, 200, 200)
        im = Image.fromarray(rgb)
        dr = ImageDraw.Draw(im)
        cx, cy = (r["x"] - x0) * s, (r["y"] - y0) * s
        dr.ellipse([cx - 6, cy - 6, cx + 6, cy + 6], outline=(255, 40, 40), width=3)
        im.save(args.png)
    print(f"{r['x']:.0f} {r['y']:.0f} {r['z']}")


if __name__ == "__main__":
    main()
