"""lattice-unroll: winding numbers straight from the organisers' trace lattice, without a spiral fit.

    lattice-unroll STORE_DIR AXIS.json Z0 Z1 OUT_DIR [--hold-mod 10 --hold-rem 7] [options]

STORE_DIR is a .vctracks folder (coordinates.i32, offsets.i64, z_bounds.i32, family_codes.i8,
metadata.json): the fitter's own input. Its traces form a lattice over every predicted sheet: "h"
traces lie inside z slices (every 8 voxels), "v" traces inside x or y planes, points 2 voxels apart,
and where a trace of one family meets one of another they share exactly the same voxel.

A scroll is one sheet wound round an axis. Give every trace one whole number x. A point p of the
trace then sits on winding  n(p) = x + sense * turns(p),  where turns(p) counts how often the trace
has gone round the axis since its first point (unwrapped angle about the published axis). Votes:
  crossing   two traces share a voxel: n must be equal there.
  next       from a point of an h trace straight outward (away from the axis, inside the slice), the
             first other trace met within --gap2 voxels is one winding further out: n + 1.
             Weight 1 up to --gap1 voxels, 0.25 beyond.
  order      the same pairs: whatever lies between, the outer one is at least one winding further out.
The whole numbers that break the least vote weight are found exactly (solver.exact: region moves by
minimum cuts). Output: one tifxyz surface per winding (rows along z, columns round the turn), named
s<number>, and numbers.npz with the number of every trace.

Traces with store index % --hold-mod == --hold-rem are never read: they are the held-out judge.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import tifffile
from scipy import sparse
from scipy.sparse import csgraph

from .solver import exact, repair

TWO_PI = 2 * np.pi


def log(*a):
    print(*a, flush=True)


def load_band(store, z0, z1, hold_mod, hold_rem, min_points, cache):
    """points of all traces that touch z0..z1, clipped to it: zyx (int32), atom of each point, family of each atom"""
    if cache and os.path.exists(cache):
        d = np.load(cache)
        return d["zyx"], d["atom"], d["fam"], d["track"]
    meta = json.load(open(os.path.join(store, "metadata.json")))
    nt, npnt = int(meta["track_count"]), int(meta["point_count"])
    off = np.fromfile(os.path.join(store, "offsets.i64"), dtype="<i8")
    zb = np.fromfile(os.path.join(store, "z_bounds.i32"), dtype="<i4").reshape(-1, 2)
    fam = np.fromfile(os.path.join(store, "family_codes.i8"), dtype=np.int8)
    coords = np.memmap(os.path.join(store, "coordinates.i32"), dtype="<i4", mode="r", shape=(npnt, 3))
    ln = off[1:] - off[:-1]
    tid = np.arange(nt)
    keep = (ln >= 2) & (fam >= 0) & (zb[:, 1] >= z0) & (zb[:, 0] < z1)
    if hold_mod > 0:
        keep &= tid % hold_mod != hold_rem
    kt = np.flatnonzero(keep)
    log(f"store: {nt / 1e6:.2f} M traces. touching z {z0} to {z1} and not held out: {len(kt) / 1e6:.2f} M traces, {ln[kt].sum() / 1e6:.1f} M points")
    zyx, trk = [], []
    step = 200_000
    for i in range(0, len(kt), step):
        part = kt[i:i + step]
        # traces of one part lie close together in the file: read the span once, pick the rows
        a, b = int(off[part[0]]), int(off[part[-1] + 1])
        if b - a > 60_000_000:                                        # a sparse stretch: read trace by trace
            blk = np.concatenate([np.asarray(coords[int(off[t]):int(off[t + 1])]) for t in part])
        else:
            buf = np.asarray(coords[a:b])
            m = np.zeros(b - a, bool)
            st = off[part] - a
            m_idx = np.repeat(st, ln[part]) + (np.arange(int(ln[part].sum())) - np.repeat(np.cumsum(ln[part]) - ln[part], ln[part]))
            m[m_idx] = True
            blk = buf[m]
        zyx.append(blk)
        trk.append(np.repeat(part, ln[part]).astype(np.int32))
        if (i // step) % 5 == 4:
            log(f"  read {i + len(part)} of {len(kt)} traces")
    zyx = np.concatenate(zyx)
    trk = np.concatenate(trk)
    # clip to the band; a trace that leaves and comes back becomes separate pieces
    inside = (zyx[:, 0] >= z0) & (zyx[:, 0] < z1)
    seq = np.cumsum(~inside)                                         # changes whenever a point was dropped
    zyx, trk, seq = zyx[inside], trk[inside], seq[inside]
    new = np.ones(len(zyx), bool)
    new[1:] = (trk[1:] != trk[:-1]) | (seq[1:] != seq[:-1])
    atom = (np.cumsum(new) - 1).astype(np.int32)
    na = int(atom[-1]) + 1
    size = np.bincount(atom, minlength=na)
    ok = size[atom] >= min_points
    zyx, atom, trk = zyx[ok], atom[ok], trk[ok]
    new = np.ones(len(zyx), bool)
    new[1:] = atom[1:] != atom[:-1]
    atom = (np.cumsum(new) - 1).astype(np.int32)
    first = np.flatnonzero(new)
    track_of_atom = trk[first]
    na = len(first)
    # family: 0 = inside a z slice, 1 = inside an x plane, 2 = inside a y plane
    f3 = np.where(fam[track_of_atom] == 0, 0, -1).astype(np.int8)
    xmin = np.minimum.reduceat(zyx[:, 2], first); xmax = np.maximum.reduceat(zyx[:, 2], first)
    ymin = np.minimum.reduceat(zyx[:, 1], first); ymax = np.maximum.reduceat(zyx[:, 1], first)
    # the store keeps even coordinates only, and a v trace wanders between two neighbouring planes (x = 8k and 8k + 2)
    f3 = np.where((f3 < 0) & (xmax - xmin <= 2), 1, f3)
    f3 = np.where((f3 < 0) & (ymax - ymin <= 2), 2, f3).astype(np.int8)
    log(f"band: {na / 1e6:.2f} M trace pieces with {min_points}+ points, {len(zyx) / 1e6:.1f} M points; families h / vx / vy / other: "
        f"{int((f3 == 0).sum())} / {int((f3 == 1).sum())} / {int((f3 == 2).sum())} / {int((f3 < 0).sum())}")
    if cache:
        np.savez(cache, zyx=zyx, atom=atom, fam=f3, track=track_of_atom)
    return zyx, atom, f3, track_of_atom


def fill_rows(X, max_gap):
    """close nan gaps of at most max_gap nodes along each row by linear interpolation (all 3 coordinates)"""
    out = X.copy()
    ok = np.isfinite(X[..., 0])
    for r in range(X.shape[0]):
        v = np.nonzero(ok[r])[0]
        if len(v) < 2:
            continue
        for gi in np.nonzero(np.diff(v) > 1)[0]:
            a, b = v[gi], v[gi + 1]
            if b - a - 1 <= max_gap:
                f = (np.arange(a + 1, b) - a) / (b - a)
                out[r, a + 1:b] = X[r, a][None] * (1 - f)[:, None] + X[r, b][None] * f[:, None]
    return out


def turns_of(th, atom, first):
    """how many times each trace piece has gone round the axis at each of its points, counted from its first point"""
    d = np.diff(th.astype(np.float64))
    d = (d + np.pi) % TWO_PI - np.pi
    d[first[1:] - 1] = 0.0
    cum = np.concatenate([[0.0], np.cumsum(d)])
    thu = th[first][atom] + (cum - cum[first][atom])
    return np.rint((thu - th) / TWO_PI).astype(np.int16)


def tree_start(n, a, b, d, w):
    """whole numbers along the spanning tree of the best-supported relations; nodes outside the largest group get no number"""
    lo, hi = np.minimum(a, b).astype(np.int64), np.maximum(a, b).astype(np.int64)
    delta = np.where(a < b, d, -d).astype(np.int64)                  # x_hi - x_lo asked by the vote
    pk = (lo * n + hi) * 64 + np.clip(delta + 32, 0, 63)
    uk, inv = np.unique(pk, return_inverse=True)
    wk = np.bincount(inv, weights=w)
    pair, dl = uk // 64, uk % 64 - 32
    o = np.lexsort((-wk, pair))
    pair, dl, wk = pair[o], dl[o], wk[o]
    firstp = np.ones(len(pair), bool)
    firstp[1:] = pair[1:] != pair[:-1]
    tot = np.bincount(np.cumsum(firstp) - 1, weights=wk)
    net = 2 * wk[firstp] - tot
    pkey, pd = pair[firstp], dl[firstp]
    ok = net > 0
    pkey, pd, net = pkey[ok], pd[ok], net[ok]
    pi, pj = pkey // n, pkey % n
    g = sparse.coo_matrix((1.0 / net, (pi, pj)), shape=(n, n)).tocsr()
    ng, grp = csgraph.connected_components(g, directed=False)
    gsize = np.bincount(grp, minlength=ng)
    main = int(np.argmax(gsize))
    tree = csgraph.minimum_spanning_tree(g)
    tree = (tree + tree.T).tocsr()
    root = int(np.flatnonzero(grp == main)[0])
    order, pred = csgraph.breadth_first_order(tree, root, directed=False, return_predecessors=True)
    v = order[1:]
    p = pred[v]
    ek = np.minimum(p, v).astype(np.int64) * n + np.maximum(p, v)
    idx = np.searchsorted(pkey, ek)
    dv = np.where(p < v, pd[idx], -pd[idx])
    lab = np.zeros(n, np.int64)
    lab_l = lab.tolist()
    for vv, pp, dd in zip(v.tolist(), p.tolist(), dv.tolist()):
        lab_l[vv] = lab_l[pp] + dd
    lab = np.array(lab_l, np.int64)
    return lab, grp == main, ng


def export(args, zyx, th, rho, wind, okp, sense, shift, t0):
    if args.no_surfaces:
        return
    lo, hi = int(wind[okp].min()), int(wind[okp].max())
    # surfaces: one per winding, rows along z, columns round the turn (in the direction of rising winding)
    src_meta = json.load(open(os.path.join(args.meta_from, "meta.json"))) if args.meta_from else {"type": "seg", "format": "tifxyz"}
    R = int(np.ceil((args.z1 - args.z0) / args.row_step))
    row = ((zyx[:, 0] - args.z0) / args.row_step).astype(np.int32)
    frac = (th.astype(np.float64) + np.pi) / TWO_PI
    if sense < 0:
        frac = 1.0 - frac
    frac = np.clip(frac, 0.0, 1.0 - 1e-9)
    order_w = np.argsort(np.where(okp, wind, hi + 1), kind="stable")
    bounds = np.searchsorted(np.where(okp, wind, hi + 1)[order_w], np.arange(lo, hi + 2))
    rep = ["sheet\tnodes\tcover\tcolumns"]
    written = 0
    masked = []
    for wn in range(lo, hi + 1):
        pi_ = order_w[bounds[wn - lo]:bounds[wn - lo + 1]]
        if len(pi_) < 500:
            continue
        W = max(16, int(round(TWO_PI * float(np.median(rho[pi_])) / args.col_step)))
        col = (frac[pi_] * W).astype(np.int64)
        cell = row[pi_] * W + col
        c_ = np.bincount(cell, minlength=R * W).astype(np.float64)
        Y = np.full((R * W, 3), np.nan, np.float32)
        got = c_ > 0
        for j, ax in enumerate((2, 1, 0)):                            # x, y, z
            Y[got, j] = (np.bincount(cell, weights=zyx[pi_, ax], minlength=R * W)[got] / c_[got]).astype(np.float32)
        Y = Y.reshape(R, W, 3)
        if args.max_spread > 0:
            # Is the cell on one sheet? Fit a flat piece through its points (distance from the axis as a plane in the
            # two grid directions, so the tilt of the sheet across the cell does not count) and look at what is left
            # over. Points of one sheet leave a voxel or two. Where points of several sheets carry this winding
            # number (crushed or folded papyrus) they leave much more, and the mean of them lies in no sheet at all.
            du = frac[pi_] * W - col - 0.5
            dv = (zyx[pi_, 0] - args.z0) / args.row_step - row[pi_] - 0.5
            r0 = rho[pi_].astype(np.float64) - float(np.median(rho[pi_]))
            acc = lambda wts=None: np.bincount(cell, weights=wts, minlength=R * W)
            n_, su, sv = acc(), acc(du), acc(dv)
            suu, svv, suv = acc(du * du), acc(dv * dv), acc(du * dv)
            sr, sur, svr, srr = acc(r0), acc(du * r0), acc(dv * r0), acc(r0 * r0)
            A = np.stack([np.stack([n_, su, sv], -1), np.stack([su, suu, suv], -1), np.stack([sv, suv, svv], -1)], -2)
            A = A + np.eye(3)[None] * 1e-3
            bvec = np.stack([sr, sur, svr], -1)
            beta = np.linalg.solve(A, bvec[..., None])[..., 0]
            left = np.sqrt(np.maximum(srr - (beta * bvec).sum(-1), 0.0) / np.maximum(n_, 1))
            bad = (got & ((left > args.max_spread) | (n_ < args.min_cell_points))).reshape(R, W)
            n_before = int(np.isfinite(Y[..., 0]).sum())
            Y[bad] = np.nan
            if not args.only or (wn + shift) in args.only:
                masked.append((wn + shift, n_before, int(bad.sum())))
                if args.only:
                    q = np.percentile(left[got & (n_ >= args.min_cell_points)], [10, 25, 50, 75, 90])
                    log(f"s{wn + shift:03d}: left over by the flat fit, voxels rms, 10/25/50/75/90th percentile of cells: " + " ".join(f"{v:.1f}" for v in q)
                        + f"; cells with fewer than {args.min_cell_points} points {float(((n_ < args.min_cell_points) & got).sum() / max(1, got.sum())):.3f}")
        if np.isfinite(Y[..., 0]).mean() < args.min_cover:
            continue
        if args.fill > 0:
            Y = fill_rows(Y, args.fill)
            Y = np.swapaxes(fill_rows(np.swapaxes(Y, 0, 1), args.fill), 0, 1)
        v = np.isfinite(Y[..., 0])
        if args.only and (wn + shift) not in args.only:
            continue
        name = f"s{wn + shift:03d}"
        od = os.path.join(args.out_dir, name)
        os.makedirs(od, exist_ok=True)
        for j, ax in enumerate("xyz"):
            tifffile.imwrite(os.path.join(od, f"{ax}.tif"), np.where(v, Y[..., j], -1.0).astype(np.float32))
        m2 = dict(src_meta)
        m2["uuid"] = name
        m2["source"] = "lattice-unroll: winding %d from the trace lattice, no spiral fit" % (wn + shift)
        m2["bbox"] = [[float(Y[v][:, k].min()) for k in range(3)], [float(Y[v][:, k].max()) for k in range(3)]]
        m2.pop("area_vx2", None); m2.pop("area_cm2", None)
        json.dump(m2, open(os.path.join(od, "meta.json"), "w"), indent=1)
        rep.append(f"{name}\t{int(v.sum())}\t{float(v.mean()):.3f}\t{W}")
        written += 1
    with open(os.path.join(args.out_dir, "export_report.tsv"), "w") as fh:
        fh.write("\n".join(rep) + "\n")
    if masked:
        tot = sum(m[1] for m in masked)
        log(f"cells taken out as not lying in one sheet: {sum(m[2] for m in masked) / max(1, tot):.3f} of {tot} (more than {args.max_spread:g} voxels left over by a flat fit, or fewer than {args.min_cell_points} points)")
    log(f"surfaces written: {written}; {time.time() - t0:.0f} s")


def inner_axis(args, zyx, th, wind, okp):
    """Where the innermost complete winding sits, slab by slab: the axis for the next band along the scroll.

    In every slab of --axis-slab slices, the lowest-numbered winding whose points cover at least 90 percent of
    the directions round the current axis. Its middle (mean over 5-degree sectors, so dense stretches do not
    pull) lies inside it. Written as control points to OUT_DIR/axis_inner.json."""
    cps = []
    sec = ((th.astype(np.float64) + np.pi) / TWO_PI * 72).astype(np.int64).clip(0, 71)
    for z0 in range(args.z0, args.z1, args.axis_slab):
        m = okp & (zyx[:, 0] >= z0) & (zyx[:, 0] < z0 + args.axis_slab)
        if m.sum() < 1000:
            continue
        w, sc, x, y = wind[m], sec[m], zyx[m, 2].astype(np.float64), zyx[m, 1].astype(np.float64)
        lo = int(w.min())
        key = (w - lo) * 72 + sc
        nw = int(w.max()) - lo + 1
        cnt = np.bincount(key, minlength=nw * 72).reshape(nw, 72)
        full = np.flatnonzero((cnt > 0).sum(1) >= 65)
        if len(full) == 0:
            continue
        k = int(full[0])
        sx = np.bincount(key, weights=x, minlength=nw * 72).reshape(nw, 72)[k]
        sy = np.bincount(key, weights=y, minlength=nw * 72).reshape(nw, 72)[k]
        ok = cnt[k] > 0
        cx, cy = float((sx[ok] / cnt[k][ok]).mean()), float((sy[ok] / cnt[k][ok]).mean())
        rad = float(np.median(np.hypot(sx[ok] / cnt[k][ok] - cx, sy[ok] / cnt[k][ok] - cy)))
        cps.append({"x": cx, "y": cy, "z": int(z0 + args.axis_slab // 2), "score": 100, "winding": k + lo, "radius": rad})
    json.dump({"control_points": cps, "source": "lattice-unroll: middle of the innermost complete winding"},
              open(os.path.join(args.out_dir, "axis_inner.json"), "w"), indent=1)
    if cps:
        log("innermost complete winding, per slab (z, x, y, median radius): " + "; ".join(f"{c['z']} {c['x']:.0f} {c['y']:.0f} r{c['radius']:.0f}" for c in cps))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("store")
    ap.add_argument("umbilicus")
    ap.add_argument("z0", type=int)
    ap.add_argument("z1", type=int)
    ap.add_argument("out_dir")
    ap.add_argument("--hold-mod", type=int, default=10)
    ap.add_argument("--hold-rem", type=int, default=7)
    ap.add_argument("--min-points", type=int, default=6)
    ap.add_argument("--sense", type=int, default=0, help="+1 or -1: windings go outward with rising or falling angle. 0 = read it from traces that go all the way round")
    ap.add_argument("--repair", type=int, default=30, help="cheap clean-up sweeps before the exact solver")
    ap.add_argument("--steps", type=int, nargs="+", default=[1], help="move sizes for the exact solver, largest first, ending in 1")
    ap.add_argument("--tail-rel", type=float, default=0.0, help="stop the exact solver when a move gains less than this share of the broken weight (0 = run to the exact minimum)")
    ap.add_argument("--cache", help="file for the band's points (default: inside OUT_DIR)")
    ap.add_argument("--pitch", type=float, default=0.0, help="radial distance per winding for the spiral start (0 = 1.25 times the median gap)")
    ap.add_argument("--split-rounds", type=int, default=0, help="rounds of: cut traces where their crossings show a step to another sheet, solve again")
    ap.add_argument("--w-cross", type=float, default=4.0)
    ap.add_argument("--w-near", type=float, default=0.0, help="weight of a near miss between an h and a v trace (0 = not used)")
    ap.add_argument("--w-next", type=float, default=1.0)
    ap.add_argument("--w-order", type=float, default=1.0)
    ap.add_argument("--gap1", type=float, default=26.0)
    ap.add_argument("--gap2", type=float, default=45.0)
    ap.add_argument("--ray-every", type=int, default=2, help="cast an outward ray from every n-th point of an h trace")
    ap.add_argument("--row-step", type=float, default=16.0)
    ap.add_argument("--col-step", type=float, default=20.0)
    ap.add_argument("--min-cover", type=float, default=0.1)
    ap.add_argument("--fill", type=int, default=2, help="close holes up to this many nodes in the output surfaces")
    ap.add_argument("--max-spread", type=float, default=0.0, help="take out cells whose points lie more than this many voxels (rms) off a flat piece fitted through them (0 = keep all)")
    ap.add_argument("--min-cell-points", type=int, default=4, help="with --max-spread: cells with fewer points are taken out")
    ap.add_argument("--meta-from", help="a tifxyz folder whose meta.json is copied to the output surfaces")
    ap.add_argument("--from-wind", help="wind.npz of a finished run on the same band and cache: skip the solve, only write surfaces")
    ap.add_argument("--only", type=int, nargs="+", help="write only these surface numbers")
    ap.add_argument("--axis-slab", type=int, default=80, help="slab height for the report of the innermost complete winding")
    ap.add_argument("--no-surfaces", action="store_true", help="solve and report only")
    args = ap.parse_args(argv)
    t0 = time.time()
    os.makedirs(args.out_dir, exist_ok=True)
    cache = args.cache or os.path.join(args.out_dir, f"band_{args.z0}_{args.z1}.npz")
    zyx, atom, fam, track_of_atom = load_band(args.store, args.z0, args.z1, args.hold_mod, args.hold_rem, args.min_points, cache)
    N, na = len(zyx), len(fam)
    first = np.flatnonzero(np.r_[True, atom[1:] != atom[:-1]])
    log(f"loaded {N / 1e6:.1f} M points, {na / 1e6:.2f} M trace pieces; {time.time() - t0:.0f} s")

    # angle about the axis and the turns each trace has made
    umb = json.load(open(args.umbilicus))
    cp = sorted(umb["control_points"], key=lambda c: c["z"])
    zs = np.array([c["z"] for c in cp], float)
    zz = np.arange(args.z0, args.z1)
    cxz = np.interp(zz, zs, [c["x"] for c in cp]).astype(np.float32)
    cyz = np.interp(zz, zs, [c["y"] for c in cp]).astype(np.float32)
    dx = zyx[:, 2] - cxz[zyx[:, 0] - args.z0]
    dy = zyx[:, 1] - cyz[zyx[:, 0] - args.z0]
    th = np.arctan2(dy, dx).astype(np.float32)
    rho = np.hypot(dx, dy).astype(np.float32)
    turn = turns_of(th, atom, first)
    log(f"turns: {int((np.maximum.reduceat(turn, first) != np.minimum.reduceat(turn, first)).sum())} trace pieces cross the seam of the angle; "
        f"radius median {float(np.median(rho[::50])):.0f}; {time.time() - t0:.0f} s")
    if args.from_wind:
        # only write the surfaces again (another grid) from the windings a finished run saved for the same band
        wd = np.load(args.from_wind)
        wind = wd["wind"].astype(np.int64)
        okp = wind > -30000
        wind = np.where(okp, wind, -10 ** 6)
        inner_axis(args, zyx, th, wind, okp)
        export(args, zyx, th, rho, wind, okp, int(wd["sense"]), int(wd["shift"]), t0)
        return

    # crossings: points of two traces of different families in the same voxel
    key = (zyx[:, 0].astype(np.int64) << 42) | (zyx[:, 1].astype(np.int64) << 21) | zyx[:, 2].astype(np.int64)
    o = np.argsort(key, kind="stable")
    ks = key[o]
    ci, cj = [], []
    for lag in (1, 2):
        same = ks[lag:] == ks[:-lag]
        i, j = o[:-lag][same], o[lag:][same]
        m = (fam[atom[i]] != fam[atom[j]]) & (atom[i] != atom[j])
        ci.append(i[m]); cj.append(j[m])
    ci, cj = np.concatenate(ci), np.concatenate(cj)
    n_exact = len(ci)
    if args.w_near > 0:
        # near misses: an h point and a v point in the same slice, one step (2 voxels) apart. The two tracers do not
        # always walk the same voxel of a band that is 5 voxels thick.
        hm = fam[atom] == 0
        hidx, vidx = np.flatnonzero(hm), np.flatnonzero(~hm)
        ov = np.argsort(key[vidx], kind="stable")
        kvs, vidx = key[vidx][ov], vidx[ov]
        taken = np.zeros(N, bool)
        taken[ci] = True; taken[cj] = True
        hidx = hidx[~taken[hidx]]
        ni_, nj_ = [], []
        for dy, dx in ((0, 2), (0, -2), (2, 0), (-2, 0)):
            kq = key[hidx] + (dy << 21) + dx
            pos = np.minimum(np.searchsorted(kvs, kq), len(kvs) - 1)
            hit = kvs[pos] == kq
            ni_.append(hidx[hit]); nj_.append(vidx[pos[hit]])
        ci = np.concatenate([ci] + ni_); cj = np.concatenate([cj] + nj_)
    w_cr = np.where(np.arange(len(ci)) < n_exact, args.w_cross, args.w_near)
    del key, o, ks
    log(f"crossings: {n_exact / 1e6:.2f} M shared voxels between traces of different families, {(len(ci) - n_exact) / 1e6:.2f} M near misses; {time.time() - t0:.0f} s")

    # outward neighbours of h traces, slice by slice, on a raster of the slice
    hp = np.flatnonzero(fam[atom] == 0)
    zo = np.argsort(zyx[hp, 0], kind="stable")
    hp = hp[zo]
    zvals, zstart = np.unique(zyx[hp, 0], return_index=True)
    zend = np.r_[zstart[1:], len(hp)]
    rr = np.arange(4.0, args.gap2 + 0.01, 0.5, dtype=np.float32)
    ni, nj, ngap = [], [], []
    self_up = self_dn = 0
    for zi in range(len(zvals)):
        idx = np.sort(hp[zstart[zi]:zend[zi]])                       # file order: consecutive points of a trace are neighbours
        x, y = zyx[idx, 2], zyx[idx, 1]
        x0, y0 = int(x.min()) - 50, int(y.min()) - 50
        W, H = int(x.max()) - x0 + 51, int(y.max()) - y0 + 51
        img = np.zeros((H, W), np.int32)
        cons = atom[idx[1:]] == atom[idx[:-1]]
        mx = ((x[1:] + x[:-1]) // 2)[cons]; my = ((y[1:] + y[:-1]) // 2)[cons]
        img[my - y0, mx - x0] = idx[:-1][cons] + 1
        img[y - y0, x - x0] = idx + 1
        thick = img.copy()
        for sy, sx in ((0, 1), (1, 0), (0, -1), (-1, 0)):
            sh = np.roll(img, (sy, sx), axis=(0, 1))
            thick = np.where(thick == 0, sh, thick)
        src = idx[::args.ray_every]
        ex, ey = np.cos(th[src]), np.sin(th[src])
        px = np.rint(zyx[src, 2][:, None] + ex[:, None] * rr[None, :]).astype(np.int32) - x0
        py = np.rint(zyx[src, 1][:, None] + ey[:, None] * rr[None, :]).astype(np.int32) - y0
        inb = (px >= 0) & (px < W) & (py >= 0) & (py < H)
        lab = np.where(inb, thick[np.clip(py, 0, H - 1), np.clip(px, 0, W - 1)], 0)
        q = np.maximum(lab - 1, 0)
        hit = (lab > 0) & ~((atom[q] == atom[src][:, None]) & (turn[q] == turn[src][:, None]))
        has = hit.any(1)
        k = hit.argmax(1)
        s_, q_, g_ = src[has], q[has, k[has]], rr[k[has]]
        selfhit = atom[q_] == atom[s_]
        dt = turn[q_[selfhit]].astype(np.int64) - turn[s_[selfhit]]
        self_up += int((dt == 1).sum()); self_dn += int((dt == -1).sum())
        ni.append(s_[~selfhit]); nj.append(q_[~selfhit]); ngap.append(g_[~selfhit])
    ni, nj, ngap = np.concatenate(ni), np.concatenate(nj), np.concatenate(ngap)
    log(f"outward neighbours: {len(ni) / 1e6:.2f} M rays met another trace (gap median {float(np.median(ngap)):.1f} voxels, {float((ngap <= args.gap1).mean()):.3f} "
        f"within {args.gap1:.0f}); a trace met itself one turn on: {self_up} times with the angle rising, {self_dn} falling; {time.time() - t0:.0f} s")
    sense = args.sense
    if sense == 0:
        if max(self_up, self_dn) < 20 or min(self_up, self_dn) > 0.2 * max(self_up, self_dn):
            raise SystemExit("cannot read the winding sense from the traces: pass --sense +1 or -1")
        sense = 1 if self_up > self_dn else -1
    log(f"winding sense {sense:+d}")

    # two points that are neighbours in space but on either side of the angle's seam: their winding indices differ by one turn
    def seam(i, j):
        return np.rint((th[j].astype(np.float64) - th[i]) / TWO_PI).astype(np.int64)

    seam_c, seam_n = seam(ci, cj), seam(ni, nj)
    w_n = np.where(ngap <= args.gap1, 1.0, 0.25) * args.w_next
    warm = None
    for rnd in range(args.split_rounds + 1):
        # votes between trace pieces
        a_c, b_c = atom[ci].astype(np.int64), atom[cj].astype(np.int64)
        d_c = sense * (turn[ci].astype(np.int64) - turn[cj] - seam_c)
        a_n, b_n = atom[ni].astype(np.int64), atom[nj].astype(np.int64)
        d_n = 1 + sense * (turn[ni].astype(np.int64) - turn[nj] - seam_n)
        a = np.concatenate([a_c, a_n]); b = np.concatenate([b_c, b_n]); d = np.concatenate([d_c, d_n])
        w = np.concatenate([w_cr, w_n])
        kind = np.concatenate([np.zeros(len(a_c), np.int8), np.ones(len(a_n), np.int8)])
        if warm is None:
            lab, inmain, ng = tree_start(na, a, b, d, w)
        else:
            ng, grp = csgraph.connected_components(sparse.coo_matrix((np.ones(len(a), np.int8), (a, b)), shape=(na, na)).tocsr(), directed=False)
            inmain = grp == int(np.argmax(np.bincount(grp, minlength=ng)))
            lab = warm
        log(f"round {rnd}: groups {ng}; the largest holds {float(inmain.mean()):.3f} of the {na / 1e6:.2f} M trace pieces and {float(inmain[atom].mean()):.3f} of the points; {time.time() - t0:.0f} s")
        # solve inside the largest group
        remap = np.full(na, -1, np.int64)
        remap[inmain] = np.arange(int(inmain.sum()))
        n = int(inmain.sum())
        mv = inmain[a] & inmain[b] & (a != b)
        a2, b2, d2, w2, k2 = remap[a[mv]], remap[b[mv]], d[mv], w[mv], kind[mv]
        uk, inv = np.unique((a2 * n + b2) * 64 + np.clip(d2 + 32, 0, 63), return_inverse=True)
        wu = np.bincount(inv, weights=w2)
        au, bu, du = (uk // 64) // n, (uk // 64) % n, uk % 64 - 32
        stacked = None
        if args.w_order > 0:
            mo = inmain[a_n] & inmain[b_n] & (a_n != b_n)
            ok_, cnt = np.unique((remap[a_n[mo]] * n + remap[b_n[mo]]) * 64 + np.clip(d_n[mo] + 32, 0, 63), return_counts=True)
            stacked = ((ok_ // 64) // n, (ok_ // 64) % n, cnt * args.w_order, ok_ % 64 - 32)
        log(f"votes: {len(uk) / 1e6:.2f} M distinct relations between {n / 1e6:.2f} M trace pieces")
        start = lab[inmain]
        if warm is None and args.repair > 0:
            start = repair(n, au, bu, du, wu, start, sweeps=args.repair, log=log)
        x = exact(n, au, bu, du, wu, start, log=log, order=stacked, steps=tuple(args.steps), tail_rel=args.tail_rel)
        res = x[b2] - x[a2] - d2
        for kk, name in ((0, "crossing"), (1, "next")):
            m = k2 == kk
            log(f"{name} votes kept: {float((res[m] == 0).mean()):.4f} of {int(m.sum()) / 1e6:.2f} M ({float(w2[m][res[m] == 0].sum() / w2[m].sum()):.4f} by weight); "
                f"asks more {float((res[m] < 0).mean()):.4f}, asks less {float((res[m] > 0).mean()):.4f}")
        xa = np.full(na, -10 ** 6, np.int64)
        xa[inmain] = x
        if rnd == args.split_rounds:
            break
        # A trace is a run along one sheet only as far as the tracer got it right. Where the traces it crosses agree
        # among themselves that it has stepped to the neighbouring sheet (a run of 3 or more crossings in a row that
        # ask for the same shift), cut it there and solve again.
        mc = mv & (kind == 0)
        rc_ = xa[b[mc]] - xa[a[mc]] - d[mc]
        nc = len(a_c)
        idx_c = np.flatnonzero(mc[:nc])
        ev_p = np.concatenate([ci[idx_c], cj[idx_c]])
        ev_ask = np.concatenate([rc_, -rc_])
        o = np.argsort(ev_p, kind="stable")
        ev_p, ev_ask = ev_p[o], ev_ask[o]
        ev_a = atom[ev_p]
        stack = []
        for sft in (-2, -1, 0, 1, 2):
            sh = np.roll(ev_ask, sft)
            same = np.roll(ev_a, sft) == ev_a
            if sft > 0:
                same[:sft] = False
            if sft < 0:
                same[sft:] = False
            stack.append(np.where(same, sh, ev_ask))
        med = np.median(np.stack(stack, 0), axis=0).astype(np.int64)
        newrun = np.ones(len(ev_p), bool)
        newrun[1:] = (ev_a[1:] != ev_a[:-1]) | (med[1:] != med[:-1])
        run = np.cumsum(newrun) - 1
        rlen = np.bincount(run)
        cut = newrun.copy()
        cut[0] = False
        cut[1:] &= (ev_a[1:] == ev_a[:-1]) & (rlen[run[1:]] >= 3) & (rlen[run[:-1]] >= 3)
        kcut = np.flatnonzero(cut)
        cut_at = (ev_p[kcut - 1] + ev_p[kcut] + 1) // 2
        flag = np.zeros(N, bool)
        flag[first] = True
        flag[cut_at] = True
        first_new = np.flatnonzero(flag)
        old_of_new = atom[first_new]
        warm = np.where(xa[old_of_new] > -10 ** 6, xa[old_of_new] + sense * turn[first_new].astype(np.int64), 0)
        # a piece cut off takes the shift its crossings asked for as its starting number
        ask_at = np.zeros(N, np.int64)
        ask_at[cut_at] = med[kcut] - med[kcut - 1]
        warm = warm + np.cumsum(ask_at)[first_new] - np.cumsum(ask_at)[first[old_of_new]]
        log(f"round {rnd}: {len(kcut)} cuts in {len(np.unique(ev_a[kcut]))} traces that step to another sheet partway")
        fam = fam[old_of_new]
        track_of_atom = track_of_atom[old_of_new]
        atom = (np.cumsum(flag) - 1).astype(np.int32)
        first = first_new
        na = len(first)
        turn = turns_of(th, atom, first)
    wind = np.where(xa[atom] > -10 ** 6, xa[atom] + sense * turn.astype(np.int64), -10 ** 6)
    okp = wind > -10 ** 6
    lo, hi = int(wind[okp].min()), int(wind[okp].max())
    cnt = np.bincount(wind[okp] - lo)
    log(f"windings {lo} to {hi}: {hi - lo + 1}; points with a number {float(okp.mean()):.3f}; windings holding 1000+ points: {int((cnt >= 1000).sum())}; {time.time() - t0:.0f} s")
    shift = 10 - lo
    np.savez(os.path.join(args.out_dir, "numbers.npz"), track=track_of_atom, number=np.where(xa > -10 ** 6, xa + shift, -1), sense=sense, shift=shift)
    np.savez(os.path.join(args.out_dir, "wind.npz"), wind=np.where(okp, wind, -30000).astype(np.int16), sense=sense, shift=shift)

    inner_axis(args, zyx, th, wind, okp)
    export(args, zyx, th, rho, wind, okp, sense, shift, t0)


if __name__ == "__main__":
    main()
