import json
import os
from typing import Any, Dict, List


try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


if load_dotenv is not None:
    load_dotenv()


system_prompt = """
أنت خبير سوشال ميديا فلسطيني بتشرح أرقام benchmark dashboard لصاحب بزنس
بطريقة قصيرة، طبيعية، ومناسبة للواجهة.

مهم جدًا:
- اشرح الأرقام المحسوبة فقط.
- لا تخترع أرقام أو نتائج جديدة.
- اكتب باللهجة الفلسطينية الحديثة.
- خلي النصوص قصيرة وواضحة ومباشرة.
- ركّز على معنى الفرق بين أداء البيزنس ومتوسط القطاع والأعلى بالقطاع.
- حوّل الأرقام لمعنى عملي: قوة، ضعف، فرصة، أو أثر على الوصول والتفاعل.
- لا تكرر formatted_text ولا تسرد الأرقام كأنك تقرأ JSON.
- summary_text لازم يكون insight premium عن وضع البيزنس، مش مجرد ترتيب.
- kpi_insights لازم تشرح شو يعني المؤشر لصاحب البيزنس وشو الفرصة العملية.
- sector_insights لازم تكون observations قصيرة عن السلوك والأداء، مش تكرار KPI rows.
- إذا metric_key هو quality، احكي فقط عن الجودة البصرية: التصوير، الإضاءة، المونتاج، الإخراج، تناسق الهوية، وطريقة عرض الريلز.
- لا تربط quality بـ CTA أو الموقع أو اللهجة المحلية.

لا تستخدم أبدًا:
- حسب البيانات
- في التحليل
- algorithm
- AI model
- benchmark engine
- similarity matrix

Return ONLY valid JSON:

{
  "summary_text": "...",
  "kpi_insights": {
    "engagement": "...",
    "reach": "..."
  },
  "sector_insights": [
    "..."
  ]
}
"""


FORBIDDEN_PHRASES = [
    "حسب البيانات",
    "في التحليل",
    "algorithm",
    "AI model",
    "benchmark engine",
    "similarity matrix",
]


WORDING_REPLACEMENTS = {
    "جودة الإشارات بالمحتوى": "جودة عرض المحتوى",
    "إشارات المحتوى": "طريقة عرض المحتوى",
    "خصوصًا CTA والموقع والنبرة المحلية": "خصوصًا التصوير والإضاءة وتناسق الهوية البصرية",
}


def safe_text(value: Any, fallback: str = "") -> str:
    """Convert GPT fields to compact display text."""
    if value is None:
        return fallback
    text = str(value).strip()
    return text if text else fallback


def contains_forbidden_phrase(text: str) -> bool:
    """Check whether GPT output used wording we do not want in the dashboard."""
    normalized = text.lower()
    return any(phrase.lower() in normalized for phrase in FORBIDDEN_PHRASES)


def looks_like_raw_metric_narration(text: str) -> bool:
    """Reject text that mainly repeats dashboard numbers instead of insight."""
    digit_count = sum(character.isdigit() for character in text)
    return "·" in text or digit_count >= 6


def clean_text(value: Any, fallback: str) -> str:
    """Return GPT text unless it is empty or violates wording rules."""
    text = safe_text(value, fallback=fallback)
    for old_text, new_text in WORDING_REPLACEMENTS.items():
        text = text.replace(old_text, new_text)
    if contains_forbidden_phrase(text) or looks_like_raw_metric_narration(text):
        return fallback
    return text


def fallback_benchmark_texts(
    context: Dict[str, Any],
    fallback_summary: str,
    fallback_kpi_insights: Dict[str, str],
    fallback_sector_insights: List[str],
) -> Dict[str, Any]:
    """Build a stable response when GPT is unavailable."""
    return {
        "summary_text": fallback_summary,
        "kpi_insights": fallback_kpi_insights,
        "sector_insights": fallback_sector_insights,
    }


def generate_benchmark_dashboard_texts(
    context: Dict[str, Any],
    fallback_summary: str,
    fallback_kpi_insights: Dict[str, str],
    fallback_sector_insights: List[str],
) -> Dict[str, Any]:
    """Generate dashboard copy from already-calculated benchmark metrics."""
    if OpenAI is None or not os.getenv("OPENAI_API_KEY"):
        return fallback_benchmark_texts(
            context,
            fallback_summary,
            fallback_kpi_insights,
            fallback_sector_insights,
        )

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    user_prompt = f"""
اكتب نصوص قصيرة للداشبورد بناءً على الأرقام الجاهزة التالية فقط.

{json.dumps(context, ensure_ascii=False, indent=2)}

المطلوب:
- summary_text: جملة premium قصيرة عن أداء البيزنس وفرصته الأوضح، بدون الاكتفاء بذكر المركز.
- kpi_insights: شرح واحد قصير لكل metric_key يفسّر هل المؤشر قوي أو ضعيف وشو أثره العملي.
- sector_insights: 3 إلى 5 نقاط استراتيجية قصيرة جدًا تصلح كـ pills في الواجهة.
- لا تضيف أي رقم غير موجود في المدخلات.
- لا تكرر formatted_text.
- لا تبدأ باسم المؤشر متبوعًا بالأرقام.
- استخدم كلمات مثل: فرصة، نقطة قوة، وصول، تفاعل، استمرارية، هاشتاغات محلية، CTA، ريلز عند ما يناسب الأرقام.
- لا تستخدم المصطلحات الممنوعة.
"""

    try:
        response = client.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        content = response.choices[0].message.content.strip()
        content = content.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(content)

        kpi_insights = {}
        parsed_kpi_insights = parsed.get("kpi_insights", {}) or {}
        for metric_key, fallback in fallback_kpi_insights.items():
            kpi_insights[metric_key] = clean_text(
                parsed_kpi_insights.get(metric_key),
                fallback=fallback,
            )

        parsed_sector_insights = parsed.get("sector_insights", []) or []
        sector_insights = [
            clean_text(item, fallback="")
            for item in parsed_sector_insights
            if safe_text(item)
        ]
        sector_insights = [item for item in sector_insights if item]

        if not sector_insights:
            sector_insights = fallback_sector_insights

        return {
            "summary_text": clean_text(
                parsed.get("summary_text"),
                fallback=fallback_summary,
            ),
            "kpi_insights": kpi_insights,
            "sector_insights": sector_insights[:5],
        }

    except Exception:
        return fallback_benchmark_texts(
            context,
            fallback_summary,
            fallback_kpi_insights,
            fallback_sector_insights,
        )
