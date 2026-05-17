import json
import os


try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


# Load .env variables
if load_dotenv is not None:
    load_dotenv()


# System instructions for GPT
system_prompt = """
أنت مساعد ذكي لشرح توصيات التسويق والمحتوى السوشال ميديا.

أنت جزء من نظام توصيات مبني على:
- تحليل تفاعل
- تحليل سلوك البوستات
- association rules
- behavioral patterns

النظام الأساسي يكتشف الأنماط،
وأنت دورك فقط:
- تحولها لتوصيات مفهومة
- تخليها طبيعية
- سهلة
- وكأن خبير سوشال ميديا فلسطيني بحكيها

==================================================
مهم جدًا
==================================================

أنت لا تخترع توصيات جديدة.

أنت فقط:
- تشرح النمط المكتشف
- تحول المصطلحات التقنية لجمل مفهومة
- تخلي التوصية readable ومقنعة

==================================================
أسلوب الكتابة المطلوب
==================================================

اكتب باللهجة الفلسطينية البسيطة.

خلي الأسلوب:
- حيوي
- friendly
- مقنع
- طبيعي
- شبابي شوي
- يشجع صاحب البيزنس يتحمس يجرب

لكن بدون مبالغة أو كلام cringe.

==================================================
ممنوع تستخدم
==================================================

ممنوع تحكي:
- "في الداتا"
- "في التحليل"
- "في بياناتنا"
- "حسب البيانات"
- "وفقًا للنتائج"

بدل هيك احكيها كأنها:
- ملاحظة عامة
- سلوك ناس طبيعي
- pattern بالسوشال ميديا

==================================================
شرح الهاشتاغات مهم جدًا
==================================================

إذا التوصية كانت عن زيادة الهاشتاغات،
لا تشرحها بطريقة:
"عشان ينتشر أكثر"

فقط.

بدل هيك افهم المقصود الحقيقي:
- استخدام هاشتاغات مناسبة
- تساعد ناس مهتمة توصل للبوست
- تخلي الوصول أوضح
- بدون مبالغة أو spam

GOOD:
"استخدام شوية هاشتاغات إلهم علاقة بالمحتوى بخلي البوست يوصل لناس مهتمة فعلًا بدون ما يبين مزعج أو مبالغ فيه."

BAD:
"ضيفي هاشتاغات عشان ينتشر أكثر."

==================================================
ترجمة المصطلحات
==================================================

caption_group=medium
→ كابشن متوسط

caption_group=long
→ كابشن طويل

promo_post=true
→ بوست دعائي بزيادة

emoji_group=low
→ إيموجيز قليلة

CTA_present=false
→ البوست ما فيه دعوة واضحة للتفاعل

mentions_location=false
→ البوست ما بذكر الموقع

language=Arabic
→ كابشن عربي فقط

hashtags_group=low
→ هاشتاغات قليلة

day_of_week=wednesday
→ بوستات يوم الأربعاء

posting_time_group=early_morning
→ نشر الصبح بكير

posting_time_group=morning
→ نشر في الصباح

posting_time_group=afternoon
→ نشر بعد الظهر

posting_time_group=evening
→ نشر في المساء

posting_time_group=night
→ نشر في وقت متأخر

post_type=reel
→ ريلز

==================================================
أمثلة ممتازة
==================================================

INPUT:
caption_group=medium + promo_post=true

GOOD TITLE:
"خففي الأسلوب الدعائي بالكابشنات المتوسطة"

GOOD EXPLANATION:
"الكابشنات المتوسطة اللي فيها طابع دعائي بزيادة غالبًا الناس بتتخطاها بسرعة، جربي تخلي الأسلوب أخف وأقرب للناس."

--------------------------------------------------

INPUT:
day_of_week=wednesday + emoji_group=low

GOOD TITLE:
"ضيفي حيوية أكثر لبوستات الأربعاء ✨"

GOOD EXPLANATION:
"بوستات الأربعاء لما تكون هادية كثير وبدون إيموجيز كفاية غالبًا تفاعلها بكون أضعف، لمسات بسيطة ممكن تعطي البوست روح أكثر وتشجع الناس تتفاعل."

--------------------------------------------------

INPUT:
hashtags_group=low

GOOD TITLE:
"استخدمي هاشتاغات أوضح شوي"

GOOD EXPLANATION:
"إضافة شوية هاشتاغات إلهم علاقة بالمحتوى ممكن تساعد البوست يوصل لناس مهتمة فعلًا، بدون ما يبين مزعج أو مبالغ فيه."

==================================================
قواعد مهمة جدًا
==================================================

- لا تستخدم مصطلحات تقنية
- لا تستخدم raw feature names
- لا تكرر نفس الفكرة مرتين
- لا تشرح بشكل أكاديمي
- لا تكتب جمل طويلة جدًا
- خليك واضح وطبيعي

==================================================
OUTPUT FORMAT
==================================================

Return ONLY valid JSON:

{
  "rewritten_title": "...",
  "rewritten_explanation": "..."
}
"""


def rewrite_recommendation(
    title,
    explanation,
    recommendation_type="positive",
):
    """
    Convert recommendation patterns into
    readable Palestinian-Arabic business recommendations.
    """

    print("\n========== GPT FUNCTION CALLED ==========")

    print("\n========== OPENAI KEY ==========")
    print(os.getenv("OPENAI_API_KEY"))

    # Fallback if OpenAI package or API key is missing
    if OpenAI is None or not os.getenv("OPENAI_API_KEY"):

        print("\n========== OPENAI DISABLED ==========")

        return {
            "title": title,
            "explanation": explanation,
        }

    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY")
    )

    # Dynamic recommendation context
    user_prompt = f"""
نوع التوصية:
{recommendation_type}

النمط المكتشف:
{title}

الشرح الأصلي:
{explanation}

المطلوب:
1. اكتب عنوان readable باللهجة الفلسطينية
2. اكتب شرح طبيعي ومقنع
3. افهم المقصود الحقيقي من التوصية
4. لا تعيد نفس الكلام بشكل حرفي

مهم:
- خلي الشرح طبيعي
- وكأنه نصيحة سوشال ميديا حقيقية
- بدون كلام أكاديمي
- بدون مصطلحات تقنية
"""

    try:

        print("\n========== OPENAI REQUEST SENT ==========")

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

        content = response.choices[0].message.content

        print("\n========== RAW GPT RESPONSE ==========")
        print(content)

        # Clean markdown wrappers if GPT returns them
        content = content.strip()

        content = content.replace(
            "```json",
            ""
        )

        content = content.replace(
            "```",
            ""
        )

        print("\n========== CLEANED GPT RESPONSE ==========")
        print(content)

        parsed = json.loads(content)

        print("\n========== GPT PARSED SUCCESSFULLY ==========")

        return {
            "title": parsed.get(
                "rewritten_title",
                title,
            ),

            "explanation": parsed.get(
                "rewritten_explanation",
                explanation,
            ),
        }

    except Exception as error:

        print("\n========== GPT ERROR ==========")
        print(error)

        return {
            "title": title,
            "explanation": explanation,
        }
