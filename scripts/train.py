"""O'qitish, benchmark va skorlash.

  python scripts/train.py                 # sintetik data: benchmark + model
  python scripts/train.py --score-uzex    # + real uzex bitimlarini lot-modeli bilan skorlash
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kartelradar.config import DATA_DIR, MODEL_DIR  # noqa: E402
from kartelradar.model import RadarModel, benchmark  # noqa: E402
from kartelradar.screens import LOT_FEATURES, build_features  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--score-uzex", action="store_true")
a = ap.parse_args()

syn = DATA_DIR / "synthetic"
lots = pd.read_csv(syn / "lots.csv")
bids = pd.read_csv(syn / "bids.csv", dtype={"firm_inn": str})
feat = build_features(lots, bids)
table, models = benchmark(feat)
print(table.to_string(index=False))
MODEL_DIR.mkdir(exist_ok=True)
table.to_csv(MODEL_DIR / "benchmark.csv", index=False)

# yakuniy modellar butun sintetik datada qayta o'qitiladi
y = feat["is_collusive"]
full = RadarModel(models["xgb_full"].features).fit_supervised(feat, y).fit_unsupervised(feat, float(y.mean()))
lot = RadarModel(LOT_FEATURES).fit_supervised(feat, y).fit_unsupervised(feat, float(y.mean()))
full.metrics, lot.metrics = models["xgb_full"].metrics, models["xgb_lot"].metrics
full.save(MODEL_DIR / "xgb_full.joblib")
lot.save(MODEL_DIR / "xgb_lot.joblib")

feat["risk"] = full.risk(feat)
feat.to_csv(DATA_DIR / "scored_synthetic.csv", index=False)
(MODEL_DIR / "metrics.json").write_text(json.dumps({"full": full.metrics, "lot": lot.metrics}, indent=2))
print(f"saqlandi: {MODEL_DIR}, {DATA_DIR / 'scored_synthetic.csv'}")

if a.score_uzex:
    u = pd.read_csv(DATA_DIR / "uzex" / "lots.csv", dtype={"customer_inn": str, "provider_inn": str, "tender_id": str})
    # real datada hudud yo'q: bozor = toifa nomi o'rniga buyurtmachi bo'yicha guruhlash yaxshiroq
    u["region"] = u["customer_inn"]
    uf = build_features(u)
    # domen siljishi: sintetik datada o'qitilgan klassifikator + real datada qayta o'qitilgan anomaliya
    lot_real = RadarModel(LOT_FEATURES).fit_unsupervised(uf, contamination=0.05)
    lot_real.clf = lot.clf
    uf["risk"] = lot_real.risk(uf)
    uf.to_csv(DATA_DIR / "scored_uzex.csv", index=False)
    lot_real.save(MODEL_DIR / "uzex_lot.joblib")
    print(f"uzex skorlandi: {len(uf)} -> {DATA_DIR / 'scored_uzex.csv'}")
