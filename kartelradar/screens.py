"""Kartel skrinlari (Imhof, Huber, Wallimann va boshq. adabiyotidagi statistik skrinlar).

Ikki daraja:
  * LOT skrinlari - faqat boshlang'ich narx, g'olib narxi, ishtirokchilar soni, buyurtmachi/g'olib
    bilan ishlaydi. Real etender.uzex.uz ma'lumotiga to'g'ridan-to'g'ri qo'llanadi.
  * BID skrinlari - har bir ishtirokchining taklif narxi kerak (CV, spread, RD, kurtosis, skewness).
    Ma'lumot bo'lmasa NaN qoladi; XGBoost NaN bilan ishlay oladi.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LOT_FEATURES = [
    "rebate", "log_start_cost", "participants_count", "single_bidder",
    "pair_share", "pair_count", "provider_market_share", "market_hhi",
    "market_n_winners", "provider_win_rate", "rotation_score",
]
BID_FEATURES = ["bid_cv", "bid_spread", "bid_rd", "bid_kurtosis", "bid_skew", "bid_diff_pct", "cover_tightness"]
ALL_FEATURES = LOT_FEATURES + BID_FEATURES

FEATURE_LABELS_UZ = {
    "rebate": "Chegirma (boshlang'ich narxdan pasayish)",
    "log_start_cost": "Lot qiymati (log)",
    "participants_count": "Ishtirokchilar soni",
    "single_bidder": "Yagona ishtirokchi",
    "pair_share": "Buyurtmachi tenderlarida shu g'olib ulushi",
    "pair_count": "Buyurtmachi-g'olib juftligi takrori",
    "provider_market_share": "G'olibning bozor ulushi",
    "market_hhi": "Bozor konsentratsiyasi (HHI)",
    "market_n_winners": "Bozordagi turli g'oliblar soni",
    "provider_win_rate": "Firmaning g'alaba darajasi",
    "rotation_score": "G'oliblar navbatlanishi (rotation)",
    "bid_cv": "Takliflar variatsiya koeffitsienti (CV)",
    "bid_spread": "Takliflar oralig'i (spread)",
    "bid_rd": "Nisbiy masofa (RD: 1-2 o'rin farqi / qolganlar dispersiyasi)",
    "bid_kurtosis": "Takliflar kurtozisi",
    "bid_skew": "Takliflar assimetriyasi",
    "bid_diff_pct": "1- va 2-o'rin orasidagi farq (%)",
    "cover_tightness": "Yutqazgan takliflar g'olibga yaqinligi",
}


def _market_key(df: pd.DataFrame) -> pd.Series:
    return df["region"].astype(str) + " | " + df["category_name"].astype(str)


def _rotation(winners: pd.Series) -> float:
    """Bozorda bir necha firma navbat bilan yutsa yuqori qiymat.
    Top-3 g'olib ulushi * (ketma-ket bir xil g'olib bo'lmaslik darajasi)."""
    w = winners.tolist()
    if len(w) < 4:
        return 0.0
    vc = pd.Series(w).value_counts(normalize=True)
    top3 = vc.iloc[:3].sum()
    switches = np.mean([a != b for a, b in zip(w, w[1:])])
    return float(top3 * switches) if 2 <= vc.size else 0.0


def lot_screens(lots: pd.DataFrame) -> pd.DataFrame:
    df = lots.copy()
    df["deal_date"] = pd.to_datetime(df["deal_date"], format="mixed")
    df = df.sort_values("deal_date")
    df["rebate"] = (1 - df["deal_cost"] / df["start_cost"]).clip(-1, 1)
    df["log_start_cost"] = np.log1p(df["start_cost"])
    df["single_bidder"] = (df["participants_count"] <= 1).astype(int)

    pair = df.groupby(["customer_inn", "provider_inn"])["tender_id"].transform("count")
    cust_n = df.groupby("customer_inn")["tender_id"].transform("count")
    df["pair_count"] = pair
    df["pair_share"] = pair / cust_n

    df["market"] = _market_key(df)
    m_n = df.groupby("market")["tender_id"].transform("count")
    pm = df.groupby(["market", "provider_inn"])["tender_id"].transform("count")
    df["provider_market_share"] = pm / m_n
    shares = df.groupby(["market", "provider_inn"]).size().div(df.groupby("market").size(), level="market")
    hhi = (shares ** 2).groupby(level="market").sum()
    df["market_hhi"] = df["market"].map(hhi)
    df["market_n_winners"] = df["market"].map(df.groupby("market")["provider_inn"].nunique())
    rot = df.groupby("market")["provider_inn"].apply(_rotation)
    df["rotation_score"] = df["market"].map(rot)
    wins = df.groupby("provider_inn")["tender_id"].count()
    df["provider_win_rate"] = df["provider_inn"].map(wins) / m_n  # oddiy proksi
    return df


def bid_screens(bids: pd.DataFrame) -> pd.DataFrame:
    def per_tender(g: pd.DataFrame) -> pd.Series:
        b = np.sort(g["bid"].to_numpy(dtype=float))
        out = dict.fromkeys(BID_FEATURES, np.nan)
        if len(b) >= 2:
            mean, sd = b.mean(), b.std(ddof=1)
            out["bid_cv"] = sd / mean if mean else np.nan
            out["bid_spread"] = (b[-1] - b[0]) / b[0] if b[0] else np.nan
            out["bid_diff_pct"] = (b[1] - b[0]) / b[0] if b[0] else np.nan
            out["cover_tightness"] = float(np.mean(b[1:] / b[0] - 1)) if b[0] else np.nan
        if len(b) >= 3:
            sd_losers = b[1:].std(ddof=1)
            out["bid_rd"] = (b[1] - b[0]) / sd_losers if sd_losers > 0 else 10.0
            s = pd.Series(b)
            out["bid_skew"] = s.skew()
        if len(b) >= 4:
            out["bid_kurtosis"] = pd.Series(b).kurt()
        return pd.Series(out)

    return bids.groupby("tender_id").apply(per_tender, include_groups=False).reset_index()


def build_features(lots: pd.DataFrame, bids: pd.DataFrame | None = None) -> pd.DataFrame:
    df = lot_screens(lots)
    if bids is not None and len(bids):
        df = df.merge(bid_screens(bids), on="tender_id", how="left")
    else:
        for c in BID_FEATURES:
            df[c] = np.nan
    return df.reset_index(drop=True)
