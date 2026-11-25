"""etender.uzex.uz dan amalga oshgan bitimlarni yig'ish (o'z kompyuteringizda ishga tushiring).

  python scripts/collect_uzex.py --max-rows 20000
  python scripts/collect_uzex.py --max-rows 5000 --participants 500   # + ishtirokchilar ro'yxati
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kartelradar.collect.uzex import collect_deals, collect_participants, normalize_deals  # noqa: E402
from kartelradar.config import DATA_DIR  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--max-rows", type=int, default=20000)
ap.add_argument("--page-size", type=int, default=500)
ap.add_argument("--pause", type=float, default=1.0, help="so'rovlar orasidagi pauza (soniya)")
ap.add_argument("--participants", type=int, default=0, help="nechta lot uchun ishtirokchilar ro'yxatini olish")
a = ap.parse_args()

raw = collect_deals(a.max_rows, a.page_size, a.pause)
lots = normalize_deals(raw)
out = DATA_DIR / "uzex"
out.mkdir(parents=True, exist_ok=True)
lots.to_csv(out / "lots.csv", index=False)
print(f"{len(lots)} ta bitim saqlandi -> {out / 'lots.csv'}")
if a.participants:
    p = collect_participants(lots["tender_id"].head(a.participants).tolist())
    print(f"ishtirokchilar -> {p}")
