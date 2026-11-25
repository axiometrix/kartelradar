"""FastAPI: uvicorn kartelradar.api.main:app --reload"""
from __future__ import annotations

import json
import os
from functools import lru_cache

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from kartelradar.config import DATA_DIR, MODEL_DIR
from kartelradar.model import RadarModel
from kartelradar.report import inspector_note, risk_level
from kartelradar.screens import build_features

DATASET = os.getenv("KR_DATASET", "synthetic")  # synthetic | uzex
app = FastAPI(title="KartelRadar API", version="0.1.0",
              description="Davlat xaridlarida til biriktirish belgilarini skrining. Natija - signal, isbot emas.")


@lru_cache
def scored() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / f"scored_{DATASET}.csv", dtype={"tender_id": str, "customer_inn": str, "provider_inn": str})
    return df.sort_values("risk", ascending=False).reset_index(drop=True)


@lru_cache
def model() -> RadarModel:
    name = "xgb_full.joblib" if DATASET == "synthetic" else "uzex_lot.joblib"
    return RadarModel.load(MODEL_DIR / name)


def _clean(d: dict) -> dict:
    return {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in d.items()}


@app.get("/health")
def health():
    return {"status": "ok", "dataset": DATASET, "rows": len(scored())}


@app.get("/metrics")
def metrics():
    p = MODEL_DIR / "metrics.json"
    bench = MODEL_DIR / "benchmark.csv"
    return {"metrics": json.loads(p.read_text()) if p.exists() else {},
            "benchmark": pd.read_csv(bench).to_dict("records") if bench.exists() else []}


@app.get("/tenders")
def tenders(min_risk: float = 0, region: str | None = None, category: str | None = None,
            limit: int = Query(50, le=500), offset: int = 0):
    df = scored()
    df = df[df["risk"] >= min_risk]
    if region:
        df = df[df["region"] == region]
    if category:
        df = df[df["category_name"] == category]
    cols = ["tender_id", "deal_date", "customer_name", "provider_name", "category_name", "region",
            "start_cost", "deal_cost", "participants_count", "rebate", "risk"]
    page = df.iloc[offset:offset + limit][cols].copy()
    page["level"] = page["risk"].map(risk_level)
    return {"total": len(df), "items": [_clean(r) for r in page.to_dict("records")]}


@app.get("/tenders/{tender_id}")
def tender(tender_id: str):
    df = scored()
    hit = df[df["tender_id"] == tender_id]
    if hit.empty:
        raise HTTPException(404, "Tender topilmadi")
    m = model()
    reasons = m.explain(hit, top_k=6)[0]
    row = _clean(hit.iloc[0].to_dict())
    return {"tender": row, "reasons": reasons, "note": inspector_note(row, reasons)}


@app.get("/firms/{inn}")
def firm(inn: str):
    df = scored()
    won = df[df["provider_inn"] == inn]
    if won.empty:
        raise HTTPException(404, "Firma topilmadi")
    return {"inn": inn, "name": won["provider_name"].iloc[0], "wins": len(won),
            "avg_risk": round(float(won["risk"].mean()), 1),
            "customers": won["customer_name"].value_counts().head(10).to_dict(),
            "top_tenders": won.head(10)[["tender_id", "risk", "deal_cost"]].to_dict("records")}


class Lot(BaseModel):
    tender_id: str
    deal_date: str
    start_cost: float
    deal_cost: float
    participants_count: int
    customer_inn: str
    customer_name: str = ""
    provider_inn: str
    provider_name: str = ""
    category_name: str = ""
    region: str = "UZ"
    bids: list[float] | None = None


@app.post("/score")
def score(lot: Lot):
    """Yangi lotni mavjud tarix kontekstida skorlash."""
    hist = scored()
    new = pd.DataFrame([lot.model_dump(exclude={"bids"})])
    base_cols = [c for c in new.columns if c in hist.columns]
    lots = pd.concat([hist[base_cols], new], ignore_index=True)
    bids = None
    if lot.bids:
        bids = pd.DataFrame({"tender_id": lot.tender_id, "firm_inn": [f"x{i}" for i in range(len(lot.bids))],
                             "bid": lot.bids, "is_winner": 0})
    feat = build_features(lots, bids)
    row = feat[feat["tender_id"] == lot.tender_id]
    m = model()
    r = float(m.risk(row)[0])
    reasons = m.explain(row, top_k=6)[0]
    d = _clean({**row.iloc[0].to_dict(), "risk": r})
    return {"risk": r, "level": risk_level(r), "reasons": reasons, "note": inspector_note(d, reasons)}
