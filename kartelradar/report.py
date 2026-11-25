"""Inspektor uchun dastlabki ma'lumotnoma (shablon). LLM ulansa, shu matnni boyitish mumkin."""
from __future__ import annotations


def risk_level(score: float) -> str:
    return "YUQORI" if score >= 70 else "O'RTA" if score >= 40 else "PAST"


def _fmt(v):
    if v is None:
        return "-"
    return f"{v:.1%}" if abs(v) < 1.5 else f"{v:,.2f}"


def inspector_note(row: dict, reasons: list[dict]) -> str:
    lines = [
        f"DASTLABKI SIGNAL (isbot emas) - tender {row.get('tender_id')}",
        f"Buyurtmachi: {row.get('customer_name')} (STIR {row.get('customer_inn')})",
        f"G'olib: {row.get('provider_name')} (STIR {row.get('provider_inn')})",
        f"Boshlang'ich narx: {row.get('start_cost', 0):,.0f} so'm; g'olib narxi: {row.get('deal_cost', 0):,.0f} so'm; "
        f"ishtirokchilar: {row.get('participants_count')}",
        f"Risk bali: {row.get('risk')} / 100 ({risk_level(row.get('risk', 0))})",
        "",
        "Asosiy sabablar (SHAP):",
    ]
    for r in reasons:
        lines.append(f"  - {r['label']}: {_fmt(r['value'])}")
    lines += ["",
              "Tavsiya: tender hujjatlari, ishtirokchilar o'rtasidagi aloqalar (ta'sischi, manzil, IP) va "
              "oldingi tenderlardagi xulq-atvorni tekshirish. Yakuniy qarorni inspektor qabul qiladi.",
              "Huquqiy asos uchun \"Raqobat to'g'risida\"gi Qonunning tegishli moddasini lex.uz rasmiy matnidan tekshiring."]
    return "\n".join(lines)
