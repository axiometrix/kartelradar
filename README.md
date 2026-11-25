# KartelRadar

Davlat xaridlarida kartel (bid rigging) belgilarini AI bilan skrining qiluvchi MVP prototipi.
*Competition & Consumer Protection AI Challenge 2026*, yo'nalish AI-10. Natija **tekshiruv uchun signal, isbot emas**.

> **Eslatma:** bu xakatondan oldingi tayyorgarlik prototipi. Tashkilotchilar oldindan yozilgan kodga ruxsat
> berishini hali tasdiqlamagan, shuning uchun xakaton boshlanishidan oldin so'rang; ruxsat bo'lmasa, g'oya va
> arxitekturani saqlab, kodni tadbir davomida qayta yozing.

## Arxitektura
```
etender.uzex.uz ochiq API ─┐                         ┌─ FastAPI  (/tenders, /tenders/{id}, /score, /firms/{inn})
sintetik generator ────────┴─> screens.py ─> model.py ┤
                              (18 skrin)  (XGBoost + IsolationForest + SHAP) └─ Streamlit dashboard
                                  └─> network.py (birga qatnashish grafi) ─> report.py (inspektor ma'lumotnomasi)
```
* **Lot skrinlari** (real datada ishlaydi): chegirma, ishtirokchilar soni, buyurtmachi-g'olib juftligi, bozor ulushi, HHI, g'oliblar navbatlanishi.
* **Bid skrinlari** (har bir taklif narxi kerak): CV, spread, RD, kurtozis, assimetriya, cover tightness.
* **Modellar:** qoida (baseline), Isolation Forest (belgisiz), XGBoost (belgili) + SHAP izohlari.

## Ishga tushirish
```bash
pip install -r requirements.txt
python scripts/make_synthetic.py --n 6000     # sintetik data (kartellar ichiga joylangan)
python scripts/train.py                       # benchmark + modellar + skorlash
streamlit run dashboard/app.py                # dashboard
uvicorn kartelradar.api.main:app --reload     # API (docs: /docs)
pytest -q
```

## Real ma'lumot (etender.uzex.uz)
```bash
python scripts/collect_uzex.py --max-rows 20000
python scripts/train.py --score-uzex          # dashboardda "etender.uzex.uz (real)"; API: KR_DATASET=uzex
```
Saytning o'z frontendi ishlatadigan ochiq endpointlar (2026-11-25 da login'siz ishladi):
`POST https://apietender.uzex.uz/api/common/DealsList` (bitimlar: boshlang'ich/g'olib narx, ishtirokchilar soni, buyurtmachi va g'olib STIR),
`GET .../api/common/GetProposalUsers/{trade_id}` (lotda qatnashgan firmalar).
Bular rasmiy hujjatlashtirilmagan: foydalanish shartlarini tekshiring, so'rovlarga pauza qo'ying.
Individual taklif narxlari ochiq emas, shuning uchun real datada faqat lot skrinlari ishlaydi.

## Muhim cheklovlar
* Sintetik datadagi metrikalar **optimistik** (kartel naqshlarini generator o'zi joylagan). Real baho uchun belgilangan
  ochiq datasetlar yoki tashkilotchi datasi kerak; real uzex datasida belgi (label) yo'q, u faqat skorlash demosi.
* Real datada "bozor" buyurtmachi INN bilan almashtirilgan (hudud maydoni yo'q).
* Huquqiy havolalar faqat lex.uz rasmiy matnidan tekshirilishi kerak.
