"""track_check.py: do output surfaces follow one sheet? Scored with the organisers' own sheet traces.

    uv run --with numpy --with scipy --with tifffile --with imagecodecs \
        python cinder/track_check.py TRACKS.npz SURF_DIR FMT --first 10 --last 129 [--z0 10000 --z1 10500]
        [--tol 6] [--up 4] [--order zyx]

TRACKS.npz is a sample from tracks_fetch.py: every track is a run of points (2 voxels apart) traced
along one predicted sheet. It is the fitter's own input, and the fitter's own score "satisfied_tracks"
asks the same question of its windings. SURF_DIR/FMT are tifxyz surfaces (fitted windings, snapped
windings or numbered sheets), one per turn, numbered outward.

A track point is on a surface when the surface passes within --tol voxels of it (distance along the
surface normal at a node within reach; surfaces are upsampled --up times first). Per track:
  covered   every point is on some surface
  one       every point is on the same surface. Surfaces n (second half of the turn) and n + 1 (first
            half) are the same sheet across the seam of the turn and count as one.
Only tracks that lie whole inside z0..z1 are scored. Printed for all tracks and for the long ones,
by tracks and by points.
"""
import argparse
import os

import numpy as np
import tifffile
from scipy import ndimage
from scipy.spatial import cKDTree


def load(d):
    P = np.stack([tifffile.imread(os.path.join(d, f"{a}.tif")).astype(np.float32) for a in "xyz"], -1)
    return P, (P[..., 0] > 0) & (P[..., 2] > 0)


def upsample(P, valid, k):
    Pz = np.where(valid[..., None], P, 0.0)
    w = ndimage.zoom(valid.astype(np.float32), k, order=1)
    up = np.stack([ndimage.zoom(Pz[..., i], k, order=1) for i in range(3)], -1)
    ok = w > 0.999
    n = np.cross(np.gradient(up, axis=1), np.gradient(up, axis=0))
    n /= np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-6)
    frac = np.broadcast_to(np.linspace(0.0, 1.0, up.shape[1], dtype=np.float32)[None, :], up.shape[:2])
    return up[ok], n[ok], frac[ok]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tracks")
    ap.add_argument("surf_dir")
    ap.add_argument("fmt")
    ap.add_argument("--first", type=int, required=True)
    ap.add_argument("--last", type=int, required=True)
    ap.add_argument("--z0", type=float, default=10000)
    ap.add_argument("--z1", type=float, default=10500)
    ap.add_argument("--tol", type=float, default=6.0)
    ap.add_argument("--up", type=int, default=4)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--order", default="zyx", help="column order of the track file")
    ap.add_argument("--only", type=int, nargs=2, metavar=("MOD", "REM"), help="score only tracks whose store index modulo MOD equals REM "
                    "(the tracks withheld from a method that takes tracks as input)")
    args = ap.parse_args()

    t = np.load(args.tracks)
    raw, st = t["xyz"], t["start"]
    X = np.stack([raw[:, args.order.index(a)] for a in "xyz"], 1).astype(np.float32)
    nt = len(st) - 1
    tr = np.repeat(np.arange(nt), np.diff(st))
    zmin = np.minimum.reduceat(X[:, 2], st[:-1])
    zmax = np.maximum.reduceat(X[:, 2], st[:-1])
    seg = np.linalg.norm(np.diff(X, axis=0), axis=1)
    seg = np.append(seg, 0.0)
    seg[st[1:] - 1] = 0.0
    length = np.add.reduceat(seg, st[:-1])
    use_t = (zmin >= args.z0) & (zmax < args.z1)
    if args.only:
        use_t &= t["track"] % args.only[0] == args.only[1]
    use_p = use_t[tr]
    X, tr = X[use_p], tr[use_p]
    print(f"tracks whole inside z {args.z0:.0f} to {args.z1:.0f}: {int(use_t.sum())} with {len(X) / 1e6:.2f} M points")

    pts, nrm, sid, frc = [], [], [], []
    pad = 40
    for k in range(args.first, args.last + 1):
        d = os.path.join(args.surf_dir, args.fmt.format(k))
        if not os.path.exists(os.path.join(d, "x.tif")):
            continue
        P, valid = load(d)
        if not valid.any():
            continue
        u, n, f = upsample(P, valid, args.up)
        m = (u[:, 2] >= args.z0 - pad) & (u[:, 2] < args.z1 + pad)
        pts.append(u[m]); nrm.append(n[m]); frc.append(f[m]); sid.append(np.full(int(m.sum()), k, np.int32))
    pts = np.concatenate(pts); nrm = np.concatenate(nrm); frc = np.concatenate(frc); sid = np.concatenate(sid)
    print(f"surfaces {args.first} to {args.last}: {len(pts) / 1e6:.1f} M nodes after upsampling {args.up} times")
    tree = cKDTree(pts)

    reach = max(args.tol, 9.0)
    pid, ida, idb = [], [], []
    for b in range(0, len(X), 2_000_000):
        xb = X[b:b + 2_000_000]
        dist, j = tree.query(xb, k=args.k, distance_upper_bound=reach, workers=-1)
        ok = np.isfinite(dist)
        jj = np.where(ok, j, 0)
        dn = np.abs(np.einsum("pkc,pkc->pk", xb[:, None, :] - pts[jj], nrm[jj]))
        ok &= dn <= args.tol
        pi, ki = np.nonzero(ok)
        s_ = sid[jj[pi, ki]]
        pid.append(pi + b); ida.append(s_); idb.append(s_ - (frc[jj[pi, ki]] < 0.5))
    pid = np.concatenate(pid); ida = np.concatenate(ida); idb = np.concatenate(idb)
    npnt = np.bincount(tr, minlength=nt)
    on_any = np.zeros(len(X), bool)
    on_any[pid] = True
    cov_pts = np.bincount(tr, weights=on_any, minlength=nt)
    covered = use_t & (cov_pts == npnt)
    best = np.zeros(nt)
    for ids in (ida, idb):
        key = np.unique(pid.astype(np.int64) * 4096 + (ids.astype(np.int64) + 8))      # one entry per point and surface
        p_, s_ = key // 4096, key % 4096
        tk, cnt = np.unique(tr[p_].astype(np.int64) * 4096 + s_, return_counts=True)
        np.maximum.at(best, tk // 4096, cnt)
    one = use_t & (best == npnt)
    print("tracks\tcount\tpoints_M\tpoints_on_a_surface\tcovered\tone_surface\tone_given_covered\tpoints_on_best_surface")
    for name, lo, hi in (("all", 0, 1e9), ("under 100 voxels", 0, 100), ("100 to 300", 100, 300), ("300 to 1000", 300, 1000), ("over 1000", 1000, 1e9)):
        m = use_t & (length >= lo) & (length < hi)
        if not m.any():
            continue
        print(f"{name}\t{int(m.sum())}\t{npnt[m].sum() / 1e6:.2f}\t{cov_pts[m].sum() / npnt[m].sum():.3f}\t{covered[m].mean():.3f}\t{one[m].mean():.3f}\t"
              f"{one[m].sum() / max(1, covered[m].sum()):.3f}\t{best[m].sum() / npnt[m].sum():.3f}")


if __name__ == "__main__":
    main()
