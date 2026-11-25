import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from kartelradar.collect.synthetic import generate
from kartelradar.model import RadarModel, benchmark
from kartelradar.screens import ALL_FEATURES, bid_screens, build_features


def test_bid_screens_known_values():
    bids = pd.DataFrame({"tender_id": ["a"] * 4, "firm_inn": list("wxyz"), "bid": [100, 101, 102, 103], "is_winner": [1, 0, 0, 0]})
    s = bid_screens(bids).iloc[0]
    assert np.isclose(s["bid_spread"], 0.03)
    assert np.isclose(s["bid_diff_pct"], 0.01)
    assert s["bid_cv"] < 0.02


def test_end_to_end_small():
    lots, bids, firms = generate(n_tenders=1500, seed=1)
    assert lots["is_collusive"].between(0, 1).all() and lots["is_collusive"].sum() > 20
    feat = build_features(lots, bids)
    assert set(ALL_FEATURES) <= set(feat.columns)
    table, models = benchmark(feat)
    xgb = table[table.model.str.startswith("XGBoost (full")].iloc[0]
    rule = table[table.model.str.startswith("Qoida")].iloc[0]
    assert xgb.pr_auc > rule.pr_auc
    m = models["xgb_full"]
    ex = m.explain(feat.head(3))
    assert len(ex) == 3


def test_lot_only_features_handle_missing_bids():
    lots, _, _ = generate(n_tenders=500, seed=2)
    feat = build_features(lots.drop(columns=["is_collusive"]))
    m = RadarModel(["rebate", "participants_count", "pair_share"]).fit_unsupervised(feat)
    r = m.risk(feat)
    assert r.min() >= 0 and r.max() <= 100


def test_api_smoke():
    from kartelradar.api.main import app
    c = TestClient(app)
    assert c.get("/health").status_code == 200
    items = c.get("/tenders", params={"limit": 5}).json()["items"]
    assert len(items) == 5
    d = c.get(f"/tenders/{items[0]['tender_id']}").json()
    assert "DASTLABKI SIGNAL" in d["note"]


def test_uzex_normalize_real_sample():
    from pathlib import Path
    from kartelradar.collect.uzex import normalize_deals
    df = normalize_deals(Path(__file__).parent / "fixtures" / "uzex_deals_sample.json")
    assert len(df) == 2 and df["tender_id"].tolist() == ["524389", "525576"]
    assert np.isclose(1 - df.deal_cost[0] / df.start_cost[0], 0.1268, atol=1e-3)
