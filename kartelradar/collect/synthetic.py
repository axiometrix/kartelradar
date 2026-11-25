"""O'zbek davlat xaridlariga o'xshash sintetik tender ma'lumoti, ichiga kartellar joylangan.

Kalibrovka: etender.uzex.uz "Amalga oshgan bitimlar" dan 2 002 ta bitim (2026-10-14 .. 2026-11-25):
  * ishtirokchilar soni: 1 -> 5%, 2 -> 49%, 3 -> 22%, 4 -> 11%, 5 -> 7%, 6+ -> 7%
  * chegirma (1 - g'olib narxi / boshlang'ich narx): median 5.0%, p25 1.0%, p75 16.7%;
    18% bitimlarda chegirma 0.5% dan kam
  * boshlang'ich narx medianasi ~173 mln so'm

Kartel xulq-atvori (OECD bid-rigging qo'llanmasidagi naqshlar):
  * bid rotation  - g'olib kartel a'zolari orasida navbat bilan almashadi
  * cover bidding - qolgan a'zolar g'olibdan biroz yuqori "soxta" taklif beradi (past dispersiya)
  * bozor taqsimoti - kartel ma'lum hudud + toifada ishlaydi
  * past chegirma - g'olib narxi boshlang'ich narxga juda yaqin
"""
from __future__ import annotations

import zlib

import numpy as np
import pandas as pd

REGIONS = ["Toshkent sh.", "Toshkent vil.", "Samarqand", "Buxoro", "Farg'ona", "Andijon", "Namangan",
           "Qashqadaryo", "Surxondaryo", "Xorazm", "Navoiy", "Jizzax", "Sirdaryo", "Qoraqalpog'iston"]
CATEGORIES = ["Qurilish-ta'mirlash ishlari", "Maktab mebellari", "Tibbiy reagentlar", "Kompyuter texnikasi",
              "Oziq-ovqat mahsulotlari", "Yoqilg'i", "Quyosh panellari", "Kanselyariya", "Yo'l qurilishi",
              "Xizmat ko'rsatish"]
PART_P = {1: .05, 2: .49, 3: .22, 4: .11, 5: .07, 6: .03, 7: .015, 8: .01, 9: .005}
FIRM_SUFFIX = ["MCHJ", "MChJ", "XK", "AJ", "OK"]
SYLL = ["NUR", "SAN", "TEX", "BUN", "YOD", "QUR", "SER", "AL", "MEGA", "STROY", "INVEST", "PRO", "LUX", "OLTIN",
        "BARAKA", "ZAMON", "IMKON", "SAHRO", "TOG", "AZIZ", "OMAD", "ELEKTRO", "MED", "AGRO"]


def _name(rng, used):
    while True:
        n = f'"{rng.choice(SYLL)}{rng.choice(SYLL)}{rng.choice(["", " GROUP", " SERVIS", " TRADE"])}" {rng.choice(FIRM_SUFFIX)}'
        if n not in used:
            used.add(n)
            return n


def generate(n_tenders: int = 6000, n_firms: int = 400, n_customers: int = 120, n_cartels: int = 6,
             rig_prob: float = 0.55, seed: int = 42):
    rng = np.random.default_rng(seed)
    used: set[str] = set()
    firms = pd.DataFrame({
        "firm_inn": [str(300000000 + i * 7919 % 9999999) for i in range(n_firms)],
        "firm_name": [_name(rng, used) for _ in range(n_firms)],
        "home_region": rng.choice(REGIONS, n_firms),
        "category": rng.choice(CATEGORIES, n_firms),
        "efficiency": rng.uniform(0.70, 0.97, n_firms),  # tannarx / boshlang'ich narx
    })
    customers = pd.DataFrame({
        "customer_inn": [str(200000000 + i * 104729 % 9999999) for i in range(n_customers)],
        "customer_name": [f"{rng.choice(REGIONS)} {rng.choice(['hokimligi', 'tibbiyot birlashmasi', 'XTB', 'qurilish boshqarmasi', 'kommunal xizmati'])} #{i}" for i in range(n_customers)],
        "region": rng.choice(REGIONS, n_customers),
    })

    # kartellar: har biri bitta hudud + toifa bozorida 3-5 a'zodan iborat
    cartels = []
    member_of = {}
    pool = list(range(n_firms))
    rng.shuffle(pool)
    for c in range(n_cartels):
        size = int(rng.integers(3, 6))
        members = [pool.pop() for _ in range(size)]
        region, cat = REGIONS[c % len(REGIONS)], CATEGORIES[(c * 3) % len(CATEGORIES)]
        firms.loc[members, ["home_region", "category"]] = [region, cat]
        for m in members:
            member_of[m] = c
        cartels.append({"cartel_id": c, "region": region, "category": cat, "members": members, "turn": 0})

    lots, bids = [], []
    start_date = pd.Timestamp("2025-01-01")
    for t in range(n_tenders):
        cust = customers.iloc[int(rng.integers(n_customers))]
        region = cust["region"]
        cat = CATEGORIES[int(rng.integers(len(CATEGORIES)))]
        # kartel bozorlariga ko'proq tender tushsin (aks holda signal juda siyrak)
        mkt = [c for c in cartels if c["region"] == region and c["category"] == cat]
        if not mkt and rng.random() < 0.10:
            c = cartels[int(rng.integers(n_cartels))]
            region, cat, mkt = c["region"], c["category"], [c]
            cust = customers[customers["region"] == region].iloc[0] if (customers["region"] == region).any() else cust
        start_cost = float(np.round(rng.lognormal(np.log(1.73e8), 1.1), -3))
        tid = f"SYN{t:06d}"
        date = start_date + pd.Timedelta(minutes=int(rng.integers(0, 60 * 24 * 690)))

        local = firms.index[(firms["category"] == cat)].to_numpy()
        rigged = bool(mkt) and rng.random() < rig_prob
        k = int(rng.choice(list(PART_P), p=np.array(list(PART_P.values())) / sum(PART_P.values())))

        rows = []
        if rigged:
            c = mkt[0]
            mem = c["members"]
            winner = mem[c["turn"] % len(mem)]
            c["turn"] += 1
            covers = [m for m in mem if m != winner]
            n_cover = int(rng.integers(1, len(covers) + 1))
            # ba'zi kartellar ehtiyotkor: chegirmani 'normal' ko'rinishga keltiradi
            win_bid = start_cost * (1 - (rng.uniform(0.0, 0.03) if rng.random() < 0.7 else rng.uniform(0.03, 0.12)))
            rows.append((winner, win_bid))
            for m in rng.choice(covers, n_cover, replace=False):
                rows.append((int(m), min(start_cost, win_bid * (1 + rng.uniform(0.003, 0.06)))))
            # tashqi raqobatchilar kamroq qatnashadi va qimmatroq taklif beradi
            if rng.random() < 0.3:
                outsiders = [o for o in local if o not in mem]
                if outsiders:
                    o = int(rng.choice(outsiders))
                    rows.append((o, min(start_cost, win_bid * (1 + rng.uniform(0.01, 0.10)))))
        else:
            cand = [f for f in local if member_of.get(f) is None or rng.random() < 0.3]
            if len(cand) < k:
                cand = list(firms.index)
            # tabiiy konsentratsiyalashgan bozor: bir nechta mahalliy firma doim qatnashadi
            if (zlib.crc32(f"{region}|{cat}".encode()) % 6 == 0) and len(cand) > 3:
                cand = sorted(cand)[:3]
            chosen = rng.choice(cand, min(k, len(cand)), replace=False)
            tight = cat in ("Kanselyariya", "Yoqilg'i", "Oziq-ovqat mahsulotlari") and rng.random() < 0.4
            for f in chosen:
                eff = firms.at[int(f), "efficiency"]
                # raqobat qancha ko'p bo'lsa, ustama shuncha past
                markup = rng.gamma(2.0, 0.06) / np.sqrt(len(chosen))
                bid = start_cost * min(1.0, eff * (1 + markup) + rng.normal(0, 0.03))
                if tight:
                    bid = start_cost * min(1.0, 0.9 + rng.normal(0, 0.01))  # standart tovar: narxlar tabiiy yaqin
                if len(chosen) <= 2 and rng.random() < 0.35:
                    bid = start_cost * (1 - rng.uniform(0, 0.01))  # raqobat yo'q, deyarli boshlang'ich narx
                rows.append((int(f), max(bid, start_cost * 0.35)))

        rows.sort(key=lambda x: x[1])
        win_firm, win_bid = rows[0]
        for i, (f, b) in enumerate(rows):
            bids.append({"tender_id": tid, "firm_inn": firms.at[f, "firm_inn"], "bid": round(b, -2), "is_winner": int(i == 0)})
        lots.append({
            "tender_id": tid, "deal_date": date, "start_cost": start_cost, "deal_cost": round(win_bid, -2),
            "participants_count": len(rows), "customer_inn": cust["customer_inn"], "customer_name": cust["customer_name"],
            "provider_inn": firms.at[win_firm, "firm_inn"], "provider_name": firms.at[win_firm, "firm_name"],
            "category_name": cat, "region": region, "is_collusive": int(rigged),
            "cartel_id": mkt[0]["cartel_id"] if rigged else -1,
        })

    lots_df = pd.DataFrame(lots).sort_values("deal_date").reset_index(drop=True)
    firm_cartel = {firms.at[m, "firm_inn"]: c["cartel_id"] for c in cartels for m in c["members"]}
    firms["cartel_id"] = firms["firm_inn"].map(firm_cartel).fillna(-1).astype(int)
    return lots_df, pd.DataFrame(bids), firms
