import json
import os
import re
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
أنت خبير سوشال ميديا فلسطيني بتشرح ليش بيزنس معيّن قريب من بيزنس ثاني،
وشو الفرق العملي بينهم بطريقة مفهومة لصاحب البيزنس.

مهم جدًا:
- أنت لا تعيد صياغة نص جاهز.
- أنت تبني الشرح من المقارنة الفعلية بين البيزنسين.
- استخدم الأرقام والسلوكيات الموجودة في المدخلات بس بدون ما تحكي بطريقة تقنية.
- لا تجعل الشرح يدور حول اسم البيزنس فقط.
- اشرح لماذا التشابه موجود: نوع المحتوى، الريلز/البوستات، النبرة، وسلوك الجمهور.
- اشرح ماذا يعمل البيزنس الأقوى بشكل أفضل: CTA، الموقع، الوصول، التعليقات، الهاشتاغات، أو تنويع المحتوى.
- اكتب باللهجة الفلسطينية الطبيعية.
- خلي الشرح قصير، واضح، ومناسب لكرت في داشبورد.
- لا تذكر كل المقاييس. اختار أهم 2 أو 3 فقط.
- الأولوية: أقوى سبب تشابه، أقوى فرق لصالح البيزنس الآخر، وخلاصة عملية قصيرة.
- تجنب الجمل الطويلة والفقرات الثقيلة.
- لا تجمع أكثر من ميزتين في نفس الشرح.
- لا تذكر engagement وviews وcomments وhashtags وCTA وlocation كلهم مع بعض.
- الأفضل تذكر: سبب تشابه واحد + فرق قوي واحد + أثر عملي واحد.
- لا تستخدم عبارة "مفيد كمقارنة" أو "مفيدة كمقارنة".
- خليك فلسطيني/شامي. تجنب المصري مثل: "وده"، "دي"، "الناس دي".
- لازم الجملة الأخيرة تكون كاملة وطبيعية، وما تنتهي بكلمة ناقصة مثل "من" أو فعل لوحده.

لا تستخدم أبدًا:
- حسب البيانات
- في التحليل
- KNN
- similarity score
- algorithm
- model
- machine learning

ركّز على:
- الأشياء المشتركة بينهم
- وين البيزنس المشابه أقوى
- شو السلوك اللي ممكن يكون مساعده
- CTA
- ذكر الموقع
- الريلز/البوستات
- التفاعل والوصول
- أسلوب المحتوى

النبرة:
- طبيعية
- حديثة
- مش أكاديمية
- مش مبالغ فيها
- قريبة من طريقة حكي خبير سوشال ميديا

Return ONLY valid JSON:

{
  "explanation": "..."
}
"""


WORDING_REPLACEMENTS = {
    "مفيدة كمقارنة": "أقوى ببعض السلوكيات",
    "مفيد كمقارنة": "أقوى ببعض السلوكيات",
    "كمقارنة مفيدة": "كنقطة مرجعية",
    "وده": "وهاد",
    "ودي": "وهاي",
    "دي ": "هاي ",
    "الناس دي": "الناس هاي",
    "بيدّيهم": "بيعطيهم",
    "بيديهم": "بيعطيهم",
    "بيقوّي": "بقوّي",
    "بيقوي": "بقوّي",
    "يساعد": "بساعد",
    "ببساعد": "بيساعد",
}


ABRUPT_ENDINGS = [
    "من",
    "في",
    "على",
    "لـ",
    "لكي",
    "عشان",
    "حتى",
    "لتستفيدوا من",
    "زيدوا",
    "حطّوا",
    "حطوا",
    "جربوا",
    "جرب",
    "جرب تخلي دعوتك",
    "خلي دعوتك",
    "دعوتك",
    "استخدموا",
    "بساعد",
    "بيساعد",
]


METRIC_STACK_TERMS = [
    "التفاعل",
    "الوصول",
    "التعليقات",
    "الهاشتاغات",
    "الموقع",
    "cta",
    "النبرة",
]


def safe_float(value: Any) -> float:
    """Convert a metric value to float for compact comparison context."""
    try:
        if value is None:
            return 0.0
        return round(float(value), 4)
    except (TypeError, ValueError):
        return 0.0


def safe_text(value: Any, fallback: str = "Unknown") -> str:
    """Convert a label value to readable text."""
    if value is None:
        return fallback
    text = str(value).strip()
    return text if text else fallback


def split_complete_sentences(text: str) -> List[str]:
    """Split text into sentences while preserving sentence punctuation."""
    parts = re.split(r"([.!؟])", text.replace("\n", " "))
    sentences: List[str] = []

    for index in range(0, len(parts), 2):
        sentence = parts[index].strip()
        punctuation = parts[index + 1] if index + 1 < len(parts) else ""
        if sentence and punctuation:
            sentences.append(f"{sentence}{punctuation}")

    return sentences


def ends_abruptly(text: str) -> bool:
    """Detect final fragments that should not be shown in frontend cards."""
    stripped = text.strip().rstrip("،,.!؟")
    if not stripped:
        return True

    if ":" in stripped:
        after_colon = stripped.rsplit(":", 1)[-1].strip()
        if len(after_colon.split()) <= 2:
            return True

    return any(stripped.endswith(ending) for ending in ABRUPT_ENDINGS)


def shorten_to_complete_sentences(text: str, fallback: str, max_words: int = 34) -> str:
    """Shorten text without cutting through the middle of a sentence."""
    words = text.split()
    if len(words) <= max_words:
        return text

    complete_sentences = split_complete_sentences(text)
    if not complete_sentences:
        return fallback

    selected: List[str] = []
    total_words = 0
    for sentence in complete_sentences:
        sentence_word_count = len(sentence.split())
        if selected and total_words + sentence_word_count > max_words:
            break
        selected.append(sentence)
        total_words += sentence_word_count
        if len(selected) == 2:
            break

    shortened = " ".join(selected).strip()
    if not shortened or ends_abruptly(shortened):
        return fallback
    return shortened


def clean_explanation_text(text: str, fallback: str) -> str:
    """Keep peer explanations compact and avoid awkward phrasing."""
    cleaned = safe_text(text, fallback=fallback)
    for old_text, new_text in WORDING_REPLACEMENTS.items():
        cleaned = cleaned.replace(old_text, new_text)

    sentences = split_complete_sentences(cleaned)
    if len(sentences) > 2:
        cleaned = " ".join(sentences[:2])

    metric_mentions = sum(
        1 for term in METRIC_STACK_TERMS
        if term.lower() in cleaned.lower()
    )
    if metric_mentions > 4 and "،" in cleaned:
        cleaned = "،".join(cleaned.split("،")[:2]).rstrip("،,.") + "."

    cleaned = shorten_to_complete_sentences(cleaned, fallback=fallback)

    if ends_abruptly(cleaned):
        return fallback

    if cleaned and cleaned[-1] not in ".!؟":
        cleaned = cleaned.rstrip("،,") + "."

    return cleaned


def describe_post_mix(profile: Dict[str, Any]) -> str:
    """Summarize whether the business leans toward reels or posts."""
    reel_share = safe_float(profile.get("pct_post_type_reel"))
    post_share = safe_float(profile.get("pct_post_type_post"))
    top_post_type = safe_text(profile.get("top_post_type"))

    if reel_share > post_share and reel_share >= 0.5:
        return "mostly reels"
    if post_share > reel_share and post_share >= 0.5:
        return "mostly regular posts"
    return top_post_type


def behavior_snapshot(profile: Dict[str, Any]) -> Dict[str, Any]:
    """Extract the business behaviors most useful for GPT explanation."""
    return {
        "business_name": safe_text(profile.get("business_name")),
        "sector": safe_text(profile.get("sector")),
        "posts_count": int(safe_float(profile.get("posts_count"))),
        "followers_count": int(safe_float(profile.get("followers_count"))),
        "content_mix": describe_post_mix(profile),
        "success_score": safe_float(profile.get("success_score")),
        "engagement_rate": safe_float(profile.get("avg_engagement_rate_followers")),
        "views_per_follower": safe_float(profile.get("avg_views_per_follower")),
        "avg_comments": safe_float(profile.get("avg_comments_count")),
        "cta_usage": safe_float(profile.get("pct_CTA_present")),
        "location_usage": safe_float(profile.get("pct_mentions_location")),
        "promo_usage": safe_float(profile.get("pct_promo_post")),
        "local_arabic_tone": safe_float(profile.get("pct_arabic_dialect_style")),
        "avg_hashtags": safe_float(profile.get("avg_hashtags_count")),
        "avg_caption_length": safe_float(profile.get("avg_caption_length")),
        "avg_emoji_count": safe_float(profile.get("avg_emoji_count")),
    }


def metric_difference(
    target: Dict[str, Any],
    peer: Dict[str, Any],
    metric: str,
    label: str,
    threshold: float = 0.10,
) -> Dict[str, Any] | None:
    """Return one meaningful peer advantage when the gap is large enough."""
    target_value = safe_float(target.get(metric))
    peer_value = safe_float(peer.get(metric))

    if peer_value <= 0:
        return None

    if peer_value > target_value * (1 + threshold):
        return {
            "behavior": label,
            "target_value": target_value,
            "peer_value": peer_value,
        }

    return None


def metric_gap(
    target: Dict[str, Any],
    peer: Dict[str, Any],
    metric: str,
    label: str,
) -> Dict[str, Any]:
    """Return the target/peer values for one comparison metric."""
    target_value = safe_float(target.get(metric))
    peer_value = safe_float(peer.get(metric))
    return {
        "behavior": label,
        "target_value": target_value,
        "peer_value": peer_value,
        "gap": round(peer_value - target_value, 4),
    }


def build_comparison_context(
    target_profile: Dict[str, Any],
    peer_profile: Dict[str, Any],
    fallback_explanation: str,
) -> Dict[str, Any]:
    """Build structured target/peer comparison data for GPT."""
    target = behavior_snapshot(target_profile)
    peer = behavior_snapshot(peer_profile)

    shared_patterns: List[str] = []
    if target["sector"] == peer["sector"]:
        shared_patterns.append(f"same sector: {target['sector']}")
    if target["content_mix"] == peer["content_mix"]:
        shared_patterns.append(f"similar content mix: {target['content_mix']}")
    if abs(target["local_arabic_tone"] - peer["local_arabic_tone"]) <= 0.15:
        shared_patterns.append("similar Arabic/local tone usage")
    if abs(target["avg_caption_length"] - peer["avg_caption_length"]) <= 20:
        shared_patterns.append("similar caption length")

    advantages = [
        item
        for item in [
            metric_difference(target, peer, "engagement_rate", "stronger engagement per follower"),
            metric_difference(target, peer, "views_per_follower", "stronger reach per follower"),
            metric_difference(target, peer, "avg_comments", "more conversation in comments"),
            metric_difference(target, peer, "cta_usage", "uses clearer calls-to-action", 0.15),
            metric_difference(target, peer, "location_usage", "mentions location more often", 0.15),
            metric_difference(target, peer, "local_arabic_tone", "uses local Arabic tone more often", 0.15),
        ]
        if item is not None
    ]

    benchmark_gaps = [
        metric_gap(target, peer, "engagement_rate", "engagement per follower"),
        metric_gap(target, peer, "views_per_follower", "reach per follower"),
        metric_gap(target, peer, "avg_comments", "comment conversation"),
        metric_gap(target, peer, "cta_usage", "CTA usage"),
        metric_gap(target, peer, "location_usage", "location mentions"),
        metric_gap(target, peer, "local_arabic_tone", "local Arabic tone"),
        metric_gap(target, peer, "avg_hashtags", "hashtag usage"),
    ]

    missing_behaviors = [
        gap for gap in benchmark_gaps
        if gap["peer_value"] > gap["target_value"] and gap["gap"] > 0
    ]

    return {
        "target_business": target,
        "peer_business": peer,
        "shared_patterns": shared_patterns,
        "stronger_peer_advantages": advantages,
        "benchmark_gaps": benchmark_gaps,
        "target_missing_behaviors": missing_behaviors,
        "existing_fallback_explanation": fallback_explanation,
    }


def generate_similar_business_explanation(
    target_profile: Dict[str, Any],
    peer_profile: Dict[str, Any],
    fallback_explanation: str,
) -> str:
    """
    Generate a fresh Palestinian-Arabic explanation from real comparison metrics.
    """
    if OpenAI is None or not os.getenv("OPENAI_API_KEY"):
        return fallback_explanation

    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY")
    )

    comparison_context = build_comparison_context(
        target_profile=target_profile,
        peer_profile=peer_profile,
        fallback_explanation=fallback_explanation,
    )

    user_prompt = f"""
اكتب شرح قصير لصاحب البيزنس يوضح ليش البيزنس المشابه قريب من أسلوبه،
وشو الأشياء اللي مخلياه أقوى أو مفيد للمقارنة.

استخدم المعلومات التالية فقط:

{json.dumps(comparison_context, ensure_ascii=False, indent=2)}

المطلوب:
- جملة واحدة أو جملتين فقط
- باللهجة الفلسطينية
- بدون مصطلحات تقنية
- بدون ذكر "حسب البيانات" أو "في التحليل"
- لا تخترع معلومات غير موجودة
- لا تكتفي بقول إن البيزنس قريب من أسلوبه.
- اذكر أهم سلوك مشترك واحد أو اثنين، وأهم فرق واحد أو اثنين عند peer_business.
- إذا كان CTA أو الموقع أو الريلز أو التفاعل أو الوصول ظاهرين كفرق، اربطهم بشكل طبيعي بالأداء.
- لا تذكر followers أو كل الأرقام إلا لو كانت ضرورية جدًا.
- لا تذكر أكثر من ميزتين أو فرقين.
- لا تسرد قائمة مقاييس.
- خليها مناسبة لكرت صغير في الواجهة.
- استخدم لهجة فلسطينية/شامية: "وهاد"، "هاي"، "بساعد"، "بخلي الناس".
- لا تستخدم لهجة مصرية مثل: "وده"، "دي"، "الناس دي"، "بيدّيهم".
- تأكد إن الجملة الأخيرة كاملة وما تنتهي بـ "من" أو "زيدوا" أو "لتستفيدوا من".
- ممنوع استخدام عبارة "مفيد كمقارنة" أو "مفيدة كمقارنة".
"""

    try:
        response = client.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
        )

        content = response.choices[0].message.content.strip()
        content = content.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(content)

        explanation = clean_explanation_text(
            parsed.get("explanation"),
            fallback=fallback_explanation,
        )
        return explanation

    except Exception:
        return fallback_explanation
