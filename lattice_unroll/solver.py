"""solver.py: whole-number labels that break the least vote weight, found exactly.

A vote (a, b, d, w) asks label[b] - label[a] = d with weight w. The cost is sum w |label[b] - label[a] - d|,
plus optional one-sided "at least" terms. This cost is convex in label differences, so it is minimised
exactly by repeated "move a set of nodes by k" steps, each found with one minimum cut (the jump moves of
Bioucas-Dias and Valadao, IEEE TIP 2007, phase unwrapping). repair() is a cheap weighted-median warm-up.
"""
import time

import numpy as np


def repair(n, a, b, d, w, lab, sweeps=30, log=print):
    """Cheap clean-up before the exact solver: every node takes the weighted median of what its votes ask for,
    half of the nodes at a time. Mends scattered single errors fast; cannot move a region (exact() does that)."""
    ts = time.time()
    lab = lab.copy()
    src = np.concatenate([a, b])
    nbr = np.concatenate([b, a])
    sgn = np.concatenate([-d, d]).astype(np.int64)
    ww = np.concatenate([w, w])
    tot_n = np.bincount(src, weights=ww, minlength=n)
    rng = np.random.default_rng(0)
    for sw in range(sweeps):
        imp = lab[nbr] + sgn
        o = np.lexsort((imp, src))
        s_src, s_imp = src[o], imp[o]
        cw = np.cumsum(ww[o])
        first = np.searchsorted(s_src, np.arange(n))
        base = np.where(first > 0, cw[np.maximum(first - 1, 0)], 0.0)
        idx = np.minimum(np.searchsorted(cw, base + 0.5 * tot_n - 1e-9), len(cw) - 1)
        med = s_imp[idx]
        ch = (tot_n > 0) & (s_src[idx] == np.arange(n)) & (med != lab) & (rng.random(n) < 0.5)
        lab[ch] = med[ch]
        if sw % 5 == 4 or ch.sum() == 0:
            res = lab[b] - lab[a] - d
            log(f"  repair sweep {sw + 1}: changed {int(ch.sum())}, vote weight kept {float(w[res == 0].sum() / w.sum()):.4f}, {time.time() - ts:.0f} s")
        if ch.sum() == 0:
            break
    return lab


def exact(n, a, b, d, w, lab, log=print, max_moves=400, tail=5.0, order=None, steps=(1,), tail_rel=0.0):
    """The exact minimum of sum w |x_b - x_a - d| over whole numbers, starting from lab.

    The repair in solve() moves one segment at a time, so a whole region that is off by one stays off by
    one: no single segment can move without breaking its strong links to its neighbours. Here one move is
    "add k to any set of segments", and the best such set is one minimum cut. For a sum of this kind
    (convex in differences) no better move with k = 1 left means the global minimum (the jump moves of
    phase unwrapping, Bioucas-Dias and Valadao 2007). Larger k first (steps = 16, 8, 4, 2, 1) only gets
    there in fewer cuts when the start is far off.

    order = (lo, hi, weight[, least]): hi is at least `least` (default 1) further out than lo. Broken,
    this costs weight times the shortfall; kept, it costs nothing (one-sided, still convex in the
    difference, so the same moves stay exact).

    One cut: for a term V(z), z the difference after the move, the four cases of (lo moves, hi moves) are
    A = V(z) for neither or both, B = V(z + k) for hi alone, C = V(z - k) for lo alone. That is a cost
    C - A on lo, A - C on hi, and B + C - 2A (never negative for convex V) when hi moves without lo."""
    import maxflow
    ts = time.time()
    lab = lab.copy()
    ids = np.arange(n)
    a = a.astype(np.int64)
    b = b.astype(np.int64)
    if order is not None:
        ha, hb, hw = order[0].astype(np.int64), order[1].astype(np.int64), order[2]
        hd = order[3] if len(order) > 3 else 1

    def energy(x):
        e_ = float((w * np.abs(x[b] - x[a] - d)).sum())
        if order is not None:
            e_ += float((hw * np.maximum(0, hd - (x[hb] - x[ha]))).sum())
        return e_

    e = energy(lab)
    e_start = e
    it = 0
    for k in steps:
        while it < max_moves:
            r = lab[b] - lab[a] - d
            A = w * np.abs(r)
            C = w * np.abs(r - k)
            cap = w * np.abs(r + k) + C - 2 * A
            un = np.bincount(a, weights=C - A, minlength=n) + np.bincount(b, weights=A - C, minlength=n)   # cost of moving
            m = cap > 0
            ea, eb, ec = [a[m]], [b[m]], [cap[m]]
            if order is not None:
                zh = lab[hb] - lab[ha] - hd
                A2 = hw * np.maximum(0, -zh)
                C2 = hw * np.maximum(0, k - zh)
                cap2 = hw * np.maximum(0, -zh - k) + C2 - 2 * A2
                un += np.bincount(ha, weights=C2 - A2, minlength=n) + np.bincount(hb, weights=A2 - C2, minlength=n)
                m2 = cap2 > 0
                ea.append(ha[m2]); eb.append(hb[m2]); ec.append(cap2[m2])
            ea, eb, ec = np.concatenate(ea), np.concatenate(eb), np.concatenate(ec)
            g = maxflow.Graph[float](n, len(ea))
            g.add_nodes(n)
            g.add_grid_tedges(ids, np.maximum(un, 0.0), np.maximum(-un, 0.0))
            g.add_edges(ea, eb, ec, np.zeros(len(ec)))
            g.maxflow()
            up = g.get_grid_segments(ids)
            new = lab + k * up
            e2 = energy(new)
            if e2 >= e - 1e-6:
                break
            gain = e - e2
            lab, e = new, e2
            it += 1
            res = lab[b] - lab[a] - d
            log(f"  exact move {it} (step {k}): {int(min(up.sum(), n - up.sum()))} segments moved, broken weight {e:.0f} (start {e_start:.0f}), "
                f"vote weight kept {float(w[res == 0].sum() / w.sum()):.4f}, {time.time() - ts:.0f} s")
            if gain < (max(tail, tail_rel * e) if k == 1 else 1e-4 * e):
                break                                                # what is left are single weak votes
    return lab


