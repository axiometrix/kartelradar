"""etender.uzex.uz ochiq "Amalga oshgan bitimlar" ro'yxatini yig'ish.

Endpoint saytning o'z frontendi ishlatadigan ochiq API (login talab qilinmaydi):
  POST {UZEX_API}/api/common/DealsList   body: {"From":1,"To":500,"currencyId":null,"System_Id":0}
  GET  {UZEX_API}/api/common/GetProposalUsers/{trade_id}   (lotga ariza/taklif bergan firmalar ro'yxati)

Eslatma:
  * Bu rasmiy hujjatlashtirilgan API emas; tuzilmasi o'zgarishi mumkin.
  * Serverga yuk bermaslik uchun so'rovlar orasida pauza bor. Foydalanish shartlarini tekshiring.
  * Individual taklif narxlari (har bir ishtirokchining bid summasi) ro'yxatda yo'q, faqat
    boshlang'ich narx, g'olib narxi va ishtirokchilar soni bor. Bid darajasidagi skrinlar
    uchun sintetik ma'lumot yoki tashkilotchi beradigan datasetdan foydalaning.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import requests

from kartelradar.config import RAW_DIR, UZEX_API

HEADERS = {"Content-Type": "application/json", "User-Agent": "KartelRadar-research/0.1"}


def fetch_deals_page(start: int, end: int, session: requests.Session, timeout: int = 60) -> list[dict]:
    body = {"From": start, "To": end, "currencyId": None, "System_Id": 0}
    r = session.post(f"{UZEX_API}/api/common/DealsList", data=json.dumps(body), headers=HEADERS, timeout=timeout)
    r.raise_for_status()
    return r.json()


def fetch_proposal_users(trade_id: int, session: requests.Session, timeout: int = 30) -> list[dict]:
    r = session.get(f"{UZEX_API}/api/common/GetProposalUsers/{trade_id}", headers=HEADERS, timeout=timeout)
    if r.status_code != 200:
        return []
    return r.json() or []


def collect_deals(max_rows: int = 20000, page_size: int = 500, pause: float = 1.0,
                  out_dir: Path = RAW_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    with requests.Session() as s:
        start = 1
        while start <= max_rows:
            end = min(start + page_size - 1, max_rows)
            page = fetch_deals_page(start, end, s)
            if not page:
                break
            rows.extend(page)
            total = page[0].get("total_count")
            print(f"  {start}-{end} yig'ildi (jami serverda: {total})")
            if total and end >= total:
                break
            start = end + 1
            time.sleep(pause)
    path = out_dir / "uzex_deals_raw.json"
    path.write_text(json.dumps(rows, ensure_ascii=False))
    return path


def collect_participants(trade_ids: list[int], pause: float = 0.5, out_dir: Path = RAW_DIR) -> Path:
    out = []
    with requests.Session() as s:
        for i, tid in enumerate(trade_ids):
            for u in fetch_proposal_users(int(tid), s):
                out.append({"tender_id": str(tid), "firm_inn": str(u.get("inn")), "firm_name": u.get("name")})
            if i % 50 == 0:
                print(f"  ishtirokchilar: {i}/{len(trade_ids)}")
            time.sleep(pause)
    path = out_dir / "uzex_participants.csv"
    pd.DataFrame(out).to_csv(path, index=False)
    return path


def normalize_deals(raw_path: Path) -> pd.DataFrame:
    """Xom JSONni KartelRadar yagona sxemasiga o'tkazadi (config.LOT_COLUMNS)."""
    df = pd.DataFrame(json.loads(Path(raw_path).read_text()))
    if df.empty:
        return df
    out = pd.DataFrame({
        "tender_id": df["trade_id"].astype(str),
        "display_no": df["display_no"].astype(str),
        "deal_date": pd.to_datetime(df["deal_date"]),
        "start_cost": df["start_cost"].astype(float),
        "deal_cost": df["deal_cost"].astype(float),
        "participants_count": df["participants_count"].astype(int),
        "customer_inn": df["customer_inn"].astype(str),
        "customer_name": df["customer_name"],
        "provider_inn": df["provider_inn"].astype(str),
        "provider_name": df["provider_name"],
        "category_name": df["category_name"].fillna("").str.strip(),
        # ro'yxatda hudud alohida maydon emas; buyurtmachi INN bozorni ifodalash uchun ishlatiladi
        "region": "UZ",
        "currency": df["currency_name"],
        "deal_status": df["deal_status_name"],
    })
    out = out[(out["start_cost"] > 0) & (out["currency"] == "Сум")]
    return out.drop_duplicates("tender_id").reset_index(drop=True)
