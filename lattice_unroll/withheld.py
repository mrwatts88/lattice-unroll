"""lattice-withheld: the withheld traces of a band, from a trace store on disk.

    lattice-withheld STORE_DIR Z0 Z1 OUT.npz [--mod 10 --rem 7] [--max-tracks 400000]

lattice-unroll never reads traces whose store index modulo --mod equals --rem (by default one in ten).
This writes exactly those (the ones that lie whole inside z0 <= z < z1) so lattice-check can score
surfaces on them: xyz (points, columns z y x), start (first point of each trace, plus the end),
track (store index).
"""
import argparse
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("store")
    ap.add_argument("z0", type=int)
    ap.add_argument("z1", type=int)
    ap.add_argument("out")
    ap.add_argument("--mod", type=int, default=10)
    ap.add_argument("--rem", type=int, default=7)
    ap.add_argument("--max-tracks", type=int, default=400000)
    args = ap.parse_args()
    meta = json.load(open(os.path.join(args.store, "metadata.json")))
    nt, npnt = int(meta["track_count"]), int(meta["point_count"])
    off = np.fromfile(os.path.join(args.store, "offsets.i64"), dtype="<i8")
    zb = np.fromfile(os.path.join(args.store, "z_bounds.i32"), dtype="<i4").reshape(-1, 2)
    coords = np.memmap(os.path.join(args.store, "coordinates.i32"), dtype="<i4", mode="r", shape=(npnt, 3))
    ln = off[1:] - off[:-1]
    sel = np.flatnonzero((ln >= 2) & (zb[:, 0] >= args.z0) & (zb[:, 1] < args.z1) & (np.arange(nt) % args.mod == args.rem))
    if len(sel) > args.max_tracks:
        sel = np.sort(np.random.default_rng(0).choice(sel, args.max_tracks, replace=False))
    xyz = np.concatenate([np.asarray(coords[int(off[t]):int(off[t + 1])]) for t in sel])
    start = np.concatenate([[0], np.cumsum(ln[sel])]).astype(np.int64)
    np.savez_compressed(args.out, xyz=xyz, start=start, track=sel.astype(np.int64))
    print(f"withheld traces whole inside z {args.z0} to {args.z1}: {len(sel)} with {len(xyz) / 1e6:.2f} M points -> {args.out}")


if __name__ == "__main__":
    main()
