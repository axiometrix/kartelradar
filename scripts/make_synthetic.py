"""Sintetik dataset yaratish: python scripts/make_synthetic.py --n 6000"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kartelradar.collect.synthetic import generate  # noqa: E402
from kartelradar.config import DATA_DIR  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=6000)
ap.add_argument("--seed", type=int, default=42)
a = ap.parse_args()
lots, bids, firms = generate(n_tenders=a.n, seed=a.seed)
out = DATA_DIR / "synthetic"
out.mkdir(parents=True, exist_ok=True)
lots.to_csv(out / "lots.csv", index=False)
bids.to_csv(out / "bids.csv", index=False)
firms.to_csv(out / "firms.csv", index=False)
print(f"{len(lots)} tender, {len(bids)} taklif, kartel ulushi {lots.is_collusive.mean():.1%} -> {out}")
