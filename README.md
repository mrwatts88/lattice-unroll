# lattice-unroll

One surface per papyrus sheet for a Herculaneum scroll, solved exactly from the public spiral-tracks
data, on a laptop, without a spiral fit and without a GPU.

**Status: draft for a Vesuvius Challenge Progress Prize entry. Not yet published.**

## What it does

The spiral fitter in `villa` (`volume-cartographer/scripts/spiral`) learns a smooth field and asks it to
keep every traced sheet ("track") at one radius. That is a soft, gradient-based use of the tracks, and on
PHerc. 343 it leaves about one winding in five or six wrong: neighbouring windings land on the same sheet,
and other sheets get no winding at all.

The same tracks can be used exactly. They form a lattice over every predicted sheet:

- `h` tracks run inside z slices (every 8 voxels) and carry a sheet round the turn,
- `v` tracks run inside x and y planes and carry it along the scroll,
- where an `h` and a `v` track meet they share the same voxel.

`lattice-unroll` gives every track one whole number, its winding. A point of the track sits on winding
`number + sense * turns`, where `turns` counts how often the track has gone round the scroll axis. Votes:

| Vote | Says | Weight |
|---|---|---|
| crossing | two tracks that share a voxel are on the same winding | 8 |
| near miss | an `h` and a `v` point one step apart are on the same winding | 4 |
| next | the first other track met going outward from an `h` point is one winding further out | 1 (0.25 beyond 26 voxels) |
| order | whatever lies between, that track is at least one winding further out | 1 |

The whole numbers that break the least vote weight are found **exactly**: the cost is convex in label
differences, so repeated "move this set of tracks by k windings" steps, each one minimum cut, reach the
global minimum (the jump moves of phase unwrapping, Bioucas-Dias and Valadão 2007). About 5 percent of
tracks step to a neighbouring sheet partway along. Their crossings disagree in a run, so the track is cut
there and the problem solved again. After two such rounds 99.3 to 99.6 percent of all crossings agree.

The output is one tifxyz surface per winding (rows along z, columns round the turn), ready for the
unchanged lasagna flatten, `vc_render_tifxyz` and the public ink models.

## Results

### PHerc. 343, against the published First Letters chain (fit + snap)

Band z 9500 to 11000. Settings fixed on a tuning band (z 10000 to 10500), then scored once on held-out
data. Withheld tracks: one in ten tracks (store index mod 10 = 7) is never read by the solver.

| Measure (held-out) | Published chain | lattice-unroll |
|---|---|---|
| Sheets on a ray with exactly one surface, slice z 9700 / 10800 | 51 / 56 of about 101 | 65 / 71 |
| Sheets on a ray with two or more surfaces | 16 / 18 | 3.5 / 2.8 |
| Withheld tracks lying on one surface, z 9600 to 9950 / 10550 to 10900 | 29 % / 43 % | 32 % / 46 % |
| Long covered withheld tracks on one surface | 79 % / 66 % | 81 % / 83 % |
| Published letters within 3 voxels of one surface | 14 of 18 | 16 of 18 |
| Ink contrast at the letters (dnative, d = 0) | 1.09 | 1.04 |
| Compute | about 3 h on an RTX 4090 | 18 min on a laptop CPU |

The known line lies on one lattice surface in one piece. On the published chain's winding it is cut in
two by a change of turn, which is why the winning submission joined two windings.

![Known line, lattice surface above, published winding below](docs/known_line_343.png)
![Papyrus fibres on the lattice surface (above) and the published winding (below)](docs/fibres_343.png)

### PHerc. 0175A, a clean unread scroll

<!-- comparison run with the published chain pending -->

| Measure | Published chain | lattice-unroll |
|---|---|---|
| Sheets on a ray with exactly one surface, slice z 8200 / 9000 | pending | 96 / 93 of 114 / 117 |
| Sheets on a ray with two or more surfaces | pending | 3 % / 3 % |
| Withheld tracks on one surface, z 8300 to 8800 | pending | 46 % |
| Compute | pending | 31 min on a laptop CPU |

## Limits

- About a fifth to a third of the sheets on a ray get no surface, as with the published chain. The gain
  is in doubled sheets made single and in sheets the fit never reached.
- The scroll axis is needed. It is placed by eye at one slice (inside the innermost loop), then carried
  along the scroll by the solver itself (`march`). A fully automatic axis is not done.
- "The next track outward is the next winding" is wrong where sheets touch or fold back. On crushed parts
  of a scroll (PHerc. 0483B, part of each turn) surfaces cut across layers there. `--max-spread` removes
  such cells by how far their points sit off a flat fit.
- Scrolls crushed into zig-zag folds with no core (PHerc. 0490B) are out of reach of this version.
- The track judge shares its tracer with the input, so it shows consistency, not truth. The independent
  checks are the known line and ink.

## Use

```bash
uv sync
# the track store of a scroll: https://dl.ash2txt.org/datasets/spiral_datasets/<SCROLL>/<volume>/tracks/
uv run lattice-unroll STORE_DIR AXIS.json Z0 Z1 OUT_DIR --w-cross 8 --w-near 4 --split-rounds 2 \
    --row-step 20 --col-step 20 --max-spread 5
```

`AXIS.json` uses the control-point layout of the published umbilicus files. The winding sense is read
from the data (tracks that go all the way round meet themselves one turn further out).

A 1,800-slice band of a large scroll needs about 16 GB of memory and 20 to 30 minutes on an M3 Pro.

## Tools

- `tools/trace_peek.py`: one cross-section of any scroll's track store, without downloading the store.
  For choosing a scroll and placing the first axis point.
- `tools/tracks_withheld.py`, `tools/track_check.py`: the withheld-track judge.

## License

MIT for the code. The scroll data used is CC BY-NC 4.0 (Vesuvius Challenge open data).
