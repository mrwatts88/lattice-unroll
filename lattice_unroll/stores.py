"""lattice-fetch: download a scroll's public trace store.

    lattice-fetch SCROLL STORES_DIR [--index-only]

SCROLL is a scroll name as on https://dl.ash2txt.org/datasets/spiral_datasets/ (for example PHerc0343),
or the full URL of a .vctracks folder. The store lands in STORES_DIR/SCROLL/: metadata.json, the three
index files (about 300 MB for a large scroll) and coordinates.i32 (9 GB for PHerc. 343). An interrupted
download resumes where it stopped. --index-only fetches the index files only (what lattice-peek needs).
"""
import argparse
import json
import os
import re
import sys
import time
import urllib.request

BASE = "https://dl.ash2txt.org/datasets/spiral_datasets"
INDEX = ("metadata.json", "family_codes.i8", "offsets.i64", "z_bounds.i32")


def get(url, rng=None):
    req = urllib.request.Request(url, headers={"Range": f"bytes={rng[0]}-{rng[1]}"} if rng else {})
    return urllib.request.urlopen(req, timeout=300).read()


def store_url(scroll):
    """URL of the scroll's .vctracks folder (the newest volume that has one)"""
    if scroll.startswith("http"):
        return scroll.rstrip("/"), scroll.rstrip("/").split("/")[-1].split("_")[0]
    vols = re.findall(r'href="(\d+)/"', get(f"{BASE}/{scroll}/").decode())
    for vol in reversed(vols):
        try:
            page = get(f"{BASE}/{scroll}/{vol}/tracks/").decode()
        except Exception:
            continue
        stores = re.findall(r'href="([^"]+\.vctracks)/"', page)
        if stores:
            return f"{BASE}/{scroll}/{vol}/tracks/{stores[0]}", scroll
    raise SystemExit(f"no trace store found for {scroll} under {BASE}/{scroll}/")


def remote_size(url):
    with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=120) as r:
        return int(r.headers["Content-Length"])


def download(url, path, size=None):
    """stream url to path, resuming a partial file; print progress"""
    if size is None:
        size = remote_size(url)
    have = os.path.getsize(path) if os.path.exists(path) else 0
    if have == size:
        return
    if have > size:
        have = 0
    req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
    with urllib.request.urlopen(req, timeout=300) as r, open(path, "ab" if have else "wb") as fh:
        total = have + int(r.headers.get("Content-Length", 0))
        t0, done, shown = time.time(), have, -1
        while True:
            buf = r.read(8 << 20)
            if not buf:
                break
            fh.write(buf)
            done += len(buf)
            pct = int(100 * done / max(total, 1))
            if total > 200e6 and pct // 5 != shown:
                shown = pct // 5
                rate = (done - have) / max(time.time() - t0, 1e-3) / 1e6
                print(f"  {os.path.basename(path)}: {done / 1e9:.2f} of {total / 1e9:.2f} GB ({pct} %), {rate:.0f} MB/s", flush=True)
    if os.path.getsize(path) != size:
        raise SystemExit(f"{path}: {os.path.getsize(path)} bytes, expected {size}. Run the command again to resume.")


def fetch(scroll, stores_dir, index_only=False):
    url, name = store_url(scroll)
    d = os.path.join(stores_dir, name)
    os.makedirs(d, exist_ok=True)
    for f in INDEX:
        download(f"{url}/{f}", os.path.join(d, f))
    meta = json.load(open(os.path.join(d, "metadata.json")))
    if not index_only:
        download(f"{url}/coordinates.i32", os.path.join(d, "coordinates.i32"), size=int(meta["point_count"]) * 12)
    return url, d, meta


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scroll")
    ap.add_argument("stores_dir")
    ap.add_argument("--index-only", action="store_true")
    args = ap.parse_args()
    url, d, meta = fetch(args.scroll, args.stores_dir, args.index_only)
    print(f"{url}\n-> {d}: {int(meta['track_count']) / 1e6:.1f} M traces, {int(meta['point_count']) / 1e6:.0f} M points", file=sys.stdout)


if __name__ == "__main__":
    main()
