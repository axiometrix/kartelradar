"""Modellar: Isolation Forest (belgisiz, real data uchun) + XGBoost (sintetik/belgili data) + SHAP."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from xgboost import XGBClassifier

from kartelradar.screens import ALL_FEATURES, FEATURE_LABELS_UZ, LOT_FEATURES


@dataclass
class RadarModel:
    features: list[str]
    clf: XGBClassifier | None = None
    iso: IsolationForest | None = None
    threshold: float = 0.5
    metrics: dict = field(default_factory=dict)

    # ---------- o'qitish ----------
    def fit_unsupervised(self, X: pd.DataFrame, contamination: float = 0.1, seed: int = 0):
        self.iso = IsolationForest(n_estimators=300, contamination=contamination, random_state=seed)
        self._iso_median = X[self.features].median()
        raw = -self.iso.fit(X[self.features].fillna(self._iso_median)).score_samples(X[self.features].fillna(self._iso_median))
        self._iso_range = (float(raw.min()), float(raw.max()))  # o'qitishdagi diapazon: bitta lot ham to'g'ri skorlanadi
        return self

    def fit_supervised(self, X: pd.DataFrame, y: pd.Series, seed: int = 0):
        pos = max(int(y.sum()), 1)
        self.clf = XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.9,
                                 colsample_bytree=0.9, scale_pos_weight=(len(y) - pos) / pos,
                                 eval_metric="aucpr", random_state=seed)
        self.clf.fit(X[self.features], y)
        return self

    # ---------- bashorat ----------
    def anomaly_score(self, X: pd.DataFrame) -> np.ndarray:
        """0..1, yuqori = g'ayrioddiy."""
        s = -self.iso.score_samples(X[self.features].fillna(self._iso_median))
        lo, hi = self._iso_range
        return np.clip((s - lo) / (hi - lo + 1e-9), 0, 1)

    def proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.clf.predict_proba(X[self.features])[:, 1]

    def risk(self, X: pd.DataFrame) -> np.ndarray:
        """Yakuniy risk 0..100. Klassifikator bo'lsa u asosiy, aks holda anomaliya bali."""
        if self.clf is not None and self.iso is not None:
            r = 0.8 * self.proba(X) + 0.2 * self.anomaly_score(X)
        elif self.clf is not None:
            r = self.proba(X)
        else:
            r = self.anomaly_score(X)
        return np.round(100 * r, 1)

    def explain(self, X: pd.DataFrame, top_k: int = 5) -> list[list[dict]]:
        """Har bir qator uchun eng katta SHAP hissali belgilar (o'zbekcha nomlar bilan)."""
        model = self.clf if self.clf is not None else self.iso
        sv = shap.TreeExplainer(model).shap_values(X[self.features].fillna(getattr(self, "_iso_median", 0))
                                                   if self.clf is None else X[self.features])
        if self.clf is None:
            sv = -sv  # IsolationForest: manfiy = anomal; belgini almashtiramiz
        out = []
        for i in range(len(X)):
            idx = np.argsort(-sv[i])[:top_k]
            out.append([{"feature": self.features[j], "label": FEATURE_LABELS_UZ.get(self.features[j], self.features[j]),
                         "value": None if pd.isna(X.iloc[i][self.features[j]]) else float(X.iloc[i][self.features[j]]),
                         "shap": float(sv[i, j])} for j in idx if sv[i, j] > 0])
        return out

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path) -> "RadarModel":
        return joblib.load(path)


def evaluate(y_true, proba, threshold=0.5) -> dict:
    pred = (np.asarray(proba) >= threshold).astype(int)
    return {"roc_auc": round(roc_auc_score(y_true, proba), 4),
            "pr_auc": round(average_precision_score(y_true, proba), 4),
            "precision": round(precision_score(y_true, pred, zero_division=0), 4),
            "recall": round(recall_score(y_true, pred, zero_division=0), 4),
            "f1": round(f1_score(y_true, pred, zero_division=0), 4),
            "threshold": threshold, "n_test": int(len(y_true)), "positives": int(np.sum(y_true))}


def benchmark(feat: pd.DataFrame, label: str = "is_collusive", seed: int = 0) -> tuple[pd.DataFrame, dict]:
    """Bozor bo'yicha guruhlangan train/test split (bir bozor ikkala tomonda bo'lmaydi -> sizib ketish kamroq).
    Qaytaradi: benchmark jadvali va o'qitilgan modellar."""
    y = feat[label].to_numpy()
    gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=seed)
    tr, te = next(gss.split(feat, y, groups=feat["market"]))
    rows, models = [], {}

    # 1) oddiy qoida (baseline): chegirma < 2% va ishtirokchilar <= 3
    rule = ((feat["rebate"] < 0.02) & (feat["participants_count"] <= 3)).astype(float).to_numpy()
    rows.append({"model": "Qoida: chegirma<2% & ishtirokchi<=3", **evaluate(y[te], rule[te])})

    # 2) Isolation Forest (belgisiz) - faqat lot belgilari
    iso = RadarModel(LOT_FEATURES).fit_unsupervised(feat.iloc[tr], contamination=float(y[tr].mean()))
    s = iso.anomaly_score(feat.iloc[te])
    thr = float(np.quantile(s, 1 - y[tr].mean()))
    rows.append({"model": "Isolation Forest (belgisiz, lot)", **evaluate(y[te], s, thr)})
    models["iso_lot"] = iso

    # 3) XGBoost lot belgilari (real uzex datasiga ko'chiriladigan)
    for name, feats in [("lot", LOT_FEATURES), ("full", ALL_FEATURES)]:
        m = RadarModel(feats).fit_supervised(feat.iloc[tr], y[tr], seed).fit_unsupervised(feat.iloc[tr], float(y[tr].mean()))
        p = m.proba(feat.iloc[te])
        m.metrics = evaluate(y[te], p)
        rows.append({"model": f"XGBoost ({name}: {len(feats)} belgi)", **m.metrics})
        models[f"xgb_{name}"] = m
    return pd.DataFrame(rows), models
