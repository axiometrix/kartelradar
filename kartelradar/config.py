from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
MODEL_DIR = ROOT / "models"

# Yagona sxema: real (uzex) va sintetik ma'lumot shu ustunlarga keltiriladi.
LOT_COLUMNS = [
    "tender_id", "deal_date", "start_cost", "deal_cost", "participants_count",
    "customer_inn", "customer_name", "provider_inn", "provider_name",
    "category_name", "region",
]
BID_COLUMNS = ["tender_id", "firm_inn", "bid", "is_winner"]

# etender.uzex.uz brauzerdagi frontend ishlatadigan ochiq endpointlar (2026-11-25 da tekshirildi)
UZEX_API = "https://apietender.uzex.uz"
