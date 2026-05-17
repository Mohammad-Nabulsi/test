import { createFileRoute } from "@tanstack/react-router";
import { motion } from "framer-motion";
import { Sparkles, ChevronLeft, MapPin, Hash, Type, Film, MessageCircle, MousePointerClick, Users, Layers, TrendingUp } from "lucide-react";

export const Route = createFileRoute("/dashboard/similar")({ component: SimilarPage });

const targetBusiness = {
  name: "بوتيك ليالي",
  category: "أزياء وستايل · رام الله",
};

const similarBusinesses = [
  { name: "نور أتلييه", handle: "@nour.atelier", similarity: 94, score: 88, note: "نفس نوع المحتوى والجمهور برام الله", color: "from-fuchsia-500 to-violet-600", initial: "ن" },
  { name: "ياسمين ستايل", handle: "@yasmin.style", similarity: 89, score: 85, note: "ريلز موضة بكابشن عربي وأسلوب قريب", color: "from-rose-500 to-pink-600", initial: "ي" },
  { name: "بيت الطرز", handle: "@bait.altarz", similarity: 86, score: 81, note: "أزياء تقليدية بلمسة عصرية — جمهور متقاطع", color: "from-amber-500 to-orange-600", initial: "ب" },
  { name: "ريم فاشن هاوس", handle: "@reem.fashion", similarity: 82, score: 77, note: "كولكشن أعراس ومناسبات بنفس الفئة السعرية", color: "from-cyan-500 to-blue-600", initial: "ر" },
  { name: "ميرا كولكشن", handle: "@mira.collection", similarity: 79, score: 74, note: "محتوى يومي وستوريهات تفاعلية", color: "from-emerald-500 to-teal-600", initial: "م" },
];

const categoryIcons: Record<string, any> = {
  "زر دعوة": MousePointerClick,
  "هاشتاجات": Hash,
  "نمط محتوى": Type,
  "ريلز": Film,
  "تفاعل": MessageCircle,
  "موقع": MapPin,
  "كابشن": Type,
  "تواصل مع الجمهور": Users,
};

const benchmarkRecommendations = [
  {
    id: 1, category: "زر دعوة", title: "ضيفي زر 'اطلبي هلأ' بنهاية كل ريل",
    why: "٤ من أصل ٥ مشاريع مشابهة بيستخدموا CTA مباشر بكل ريل، وهاد بيرفع التحويلات عندهم بـ٣١٪ بالمتوسط مقارنة فيكي.",
    impact: 34, confidence: 91, peers: ["نور أتلييه", "ياسمين ستايل", "ريم فاشن هاوس"],
  },
  {
    id: 2, category: "موقع", title: "اذكري 'رام الله' بشكل أوضح بالكابشن والهاشتاجات",
    why: "المشاريع الشبيهة بتذكر موقعها بـ٧٢٪ من بوستاتها — إنتي بس بـ٢٨٪. هاد بيوسّع الوصول للجمهور المحلي.",
    impact: 26, confidence: 87, peers: ["نور أتلييه", "بيت الطرز"],
  },
  {
    id: 3, category: "نمط محتوى", title: "نوّعي بأنواع المحتوى — مش بس عروض",
    why: "نور أتلييه وميرا كولكشن بيقسّموا محتواهم: ٤٠٪ ستايل يومي، ٣٠٪ كواليس، ٢٠٪ عروض، ١٠٪ زبائن. إنتي ٦٢٪ عروض.",
    impact: 29, confidence: 84, peers: ["نور أتلييه", "ميرا كولكشن"],
  },
  {
    id: 4, category: "ريلز", title: "جرّبي ريلز 'قبل وبعد' للتنسيق",
    why: "ياسمين ستايل وبيت الطرز بيعملوا ريل أسبوعي بصيغة قبل/بعد، ومتوسط مشاهداته ٢.٣ مرة أعلى من ريلزك الحالية.",
    impact: 41, confidence: 82, peers: ["ياسمين ستايل", "بيت الطرز"],
  },
  {
    id: 5, category: "تفاعل", title: "ردّي على الكومنتات خلال أول ساعة",
    view: true,
    why: "المشاريع المشابهة الأقوى بيردّوا خلال ٤٨ دقيقة بالمتوسط. الرد السريع بيخلّي الخوارزمية تعرض البوست أكتر.",
    impact: 22, confidence: 89, peers: ["نور أتلييه", "ياسمين ستايل"],
  },
  {
    id: 6, category: "كابشن", title: "اكتبي سؤال بنهاية كل كابشن",
    why: "ميرا كولكشن بتنهي ٨٠٪ من بوستاتها بسؤال للمتابعين، ومعدل الكومنتات عندها أعلى بـ٤٧٪.",
    impact: 19, confidence: 78, peers: ["ميرا كولكشن", "ياسمين ستايل"],
  },
  {
    id: 7, category: "هاشتاجات", title: "ضيفي هاشتاجات محلية أصغر وأكثر تخصصاً",
    why: "بيت الطرز بيستخدم هاشتاجات زي #ستايل_الخليل و#رام_الله_شوبينج — وصول أعمق وجمهور أنشط من الهاشتاجات الكبيرة.",
    impact: 17, confidence: 75, peers: ["بيت الطرز", "ريم فاشن هاوس"],
  },
  {
    id: 8, category: "تواصل مع الجمهور", title: "اعملي ستوري سؤال وجواب أسبوعي",
    why: "٤ من ٥ مشاريع شبيهة بيعملوا 'Q&A' ثابت أسبوعياً، وهاد بيرفع مدة المشاهدة بالستوري بـ٣٦٪.",
    impact: 24, confidence: 80, peers: ["نور أتلييه", "ميرا كولكشن", "ياسمين ستايل"],
  },
];

const insightHighlights = [
  "بيستخدموا CTA مباشر بكل ريل",
  "بينوّعوا بأنواع المحتوى أكتر",
  "بيذكروا الموقع بشكل واضح",
  "بيردّوا على الكومنتات بسرعة",
  "بينهوا الكابشن بسؤال",
  "بيعملوا Q&A أسبوعي بالستوري",
  "بيستخدموا هاشتاجات محلية مخصّصة",
  "بيعرضوا كواليس العمل",
];

function SimilarPage() {
  const strongest = benchmarkRecommendations.reduce((a, b) => (a.impact > b.impact ? a : b));

  return (
    <div className="space-y-6">
      {/* HERO */}
      <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="relative overflow-hidden rounded-2xl border border-border p-6 bg-mesh">
        <div className="absolute -top-20 -left-20 h-64 w-64 bg-gradient-brand opacity-20 blur-3xl rounded-full" />
        <div className="relative flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div className="flex items-start gap-3">
            <div className="h-11 w-11 rounded-xl bg-gradient-brand flex items-center justify-center shadow-glow shrink-0"><Sparkles className="h-5 w-5 text-white" /></div>
            <div>
              <h2 className="font-display text-2xl tracking-tight">خطوات مستوحاة من مشاريع قريبة من أسلوبك ✨</h2>
              <p className="text-sm text-muted-foreground mt-1 max-w-xl">حلّلنا {similarBusinesses.length} مشاريع شبيهة لـ{targetBusiness.name} ولقطنا {benchmarkRecommendations.length} توصيات قابلة للتطبيق — مرتّبة حسب الأثر المتوقع.</p>
            </div>
          </div>
          <div className="flex gap-3">
            <div className="rounded-xl border border-border bg-card/60 backdrop-blur p-3 min-w-[120px]">
              <div className="text-[10px] uppercase tracking-widest text-muted-foreground">توصيات</div>
              <div className="num-display text-2xl font-semibold mt-0.5">{benchmarkRecommendations.length}</div>
            </div>
            <div className="rounded-xl border border-[var(--brand)]/30 bg-gradient-soft p-3 min-w-[160px]">
              <div className="text-[10px] uppercase tracking-widest text-muted-foreground flex items-center gap-1"><TrendingUp className="h-3 w-3" /> أقوى فرصة</div>
              <div className="text-sm font-medium mt-0.5 line-clamp-1">{strongest.title}</div>
              <div className="text-gradient-brand num-display text-sm font-semibold">+{strongest.impact}% أثر متوقع</div>
            </div>
          </div>
        </div>
      </motion.div>

      {/* SIMILAR BUSINESSES */}
      <div>
        <div className="flex items-center justify-between mb-3">
          <div>
            <h3 className="font-display text-lg tracking-tight">مشاريع مشابهة إلك</h3>
            <p className="text-xs text-muted-foreground">مختارين حسب نوع المحتوى، الجمهور، والمنطقة</p>
          </div>
          <span className="text-[10px] uppercase tracking-widest text-muted-foreground">{similarBusinesses.length} مشاريع</span>
        </div>
        <div className="flex gap-3 overflow-x-auto pb-2 -mx-1 px-1 snap-x">
          {similarBusinesses.map((b, i) => (
            <motion.div key={b.name} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: i * 0.06 }}
              className="snap-start shrink-0 w-[260px] rounded-2xl border border-border bg-card shadow-card hover:shadow-elegant transition-all p-4">
              <div className="flex items-center gap-3">
                <div className={`h-10 w-10 rounded-lg bg-gradient-to-br ${b.color} flex items-center justify-center text-white text-sm font-semibold`}>{b.initial}</div>
                <div className="min-w-0 flex-1">
                  <div className="text-sm font-medium truncate">{b.name}</div>
                  <div className="text-[11px] text-muted-foreground truncate">{b.handle}</div>
                </div>
              </div>
              <p className="text-xs text-muted-foreground mt-3 leading-relaxed line-clamp-2 min-h-[32px]">{b.note}</p>
              <div className="mt-3 space-y-2">
                <div className="flex items-center gap-2">
                  <span className="text-[10px] text-muted-foreground w-14 shrink-0">التشابه</span>
                  <div className="flex-1 h-1.5 rounded-full bg-muted overflow-hidden">
                    <motion.div initial={{ width: 0 }} animate={{ width: `${b.similarity}%` }} transition={{ duration: 1, delay: i * 0.05 }} className="h-full bg-gradient-brand" />
                  </div>
                  <span className="text-xs num-display font-semibold w-9 text-left">{b.similarity}%</span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-[10px] text-muted-foreground w-14 shrink-0">النجاح</span>
                  <div className="flex-1 h-1.5 rounded-full bg-muted overflow-hidden">
                    <motion.div initial={{ width: 0 }} animate={{ width: `${b.score}%` }} transition={{ duration: 1, delay: i * 0.05 + 0.1 }} className="h-full bg-[var(--brand-3)]" />
                  </div>
                  <span className="text-xs num-display font-semibold w-9 text-left">{b.score}</span>
                </div>
              </div>
            </motion.div>
          ))}
        </div>
      </div>

      {/* RECOMMENDATION CARDS */}
      <div>
        <div className="flex items-center justify-between mb-3">
          <div>
            <h3 className="font-display text-lg tracking-tight">توصيات مستخرجة من سلوك المشاريع الشبيهة</h3>
            <p className="text-xs text-muted-foreground">كل توصية مدعومة بمقارنة فعلية مع مشاريع من نفس النوع</p>
          </div>
          <span className="text-[10px] uppercase tracking-widest text-muted-foreground">مرتّبة حسب الأثر</span>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          {benchmarkRecommendations.map((r, i) => {
            const Icon = categoryIcons[r.category] ?? Sparkles;
            return (
              <motion.div key={r.id} initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: i * 0.05 }} whileHover={{ y: -2 }}
                className="group relative overflow-hidden rounded-2xl border border-border bg-card shadow-card hover:shadow-elegant transition-all p-5">
                <div className="absolute top-0 left-0 h-40 w-40 bg-gradient-brand opacity-5 blur-3xl rounded-full" />
                <div className="relative flex items-start justify-between gap-4">
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-2">
                      <div className="h-8 w-8 rounded-lg bg-accent flex items-center justify-center"><Icon className="h-4 w-4 text-[var(--brand)]" /></div>
                      <span className="text-[10px] uppercase tracking-widest text-muted-foreground font-medium">{r.category}</span>
                    </div>
                    <h3 className="font-display text-lg tracking-tight">{r.title}</h3>
                    <p className="text-sm text-muted-foreground mt-3 leading-relaxed">{r.why}</p>
                  </div>
                  <div className="text-left shrink-0">
                    <div className="text-xs text-muted-foreground">الأثر</div>
                    <div className="text-gradient-brand num-display text-xl font-semibold">+{r.impact}%</div>
                  </div>
                </div>

                <div className="relative mt-4 flex items-center gap-2">
                  <span className="text-[10px] text-muted-foreground">مستوحاة من:</span>
                  <div className="flex flex-wrap gap-1">
                    {r.peers.map((p) => (
                      <span key={p} className="text-[10px] px-2 py-0.5 rounded-full bg-accent border border-border">{p}</span>
                    ))}
                  </div>
                </div>

                <div className="relative mt-4 flex items-center justify-between gap-4">
                  <div className="flex items-center gap-2 flex-1 max-w-xs">
                    <span className="text-[11px] text-muted-foreground w-14">الثقة</span>
                    <div className="flex-1 h-1.5 rounded-full bg-muted overflow-hidden">
                      <motion.div initial={{ width: 0 }} animate={{ width: `${r.confidence}%` }} transition={{ duration: 1.2, delay: i * 0.05 }} className="h-full bg-gradient-brand" />
                    </div>
                    <span className="text-xs font-medium num-display">{r.confidence}%</span>
                  </div>
                  <button className="text-xs flex items-center gap-1 px-3 py-1.5 rounded-lg border border-border hover:bg-accent transition-colors">طبّقي <ChevronLeft className="h-3 w-3" /></button>
                </div>
              </motion.div>
            );
          })}
        </div>
      </div>

      {/* INSIGHT HIGHLIGHTS */}
      <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="rounded-2xl border border-border bg-card shadow-card p-5">
        <div className="flex items-center gap-2 mb-3">
          <div className="h-8 w-8 rounded-lg bg-accent flex items-center justify-center"><Layers className="h-4 w-4 text-[var(--brand)]" /></div>
          <div>
            <h3 className="font-display text-base tracking-tight">شو المشاريع المشابهة بتعمل بشكل أفضل؟</h3>
            <p className="text-[11px] text-muted-foreground">ملخص سريع للسلوكيات اللي بتفرّقهم</p>
          </div>
        </div>
        <div className="flex flex-wrap gap-2">
          {insightHighlights.map((h, i) => (
            <motion.span key={h} initial={{ opacity: 0, scale: 0.9 }} animate={{ opacity: 1, scale: 1 }} transition={{ delay: i * 0.04 }}
              className="text-xs px-3 py-1.5 rounded-full bg-gradient-soft border border-border hover:border-[var(--brand)]/40 transition-colors">
              {h}
            </motion.span>
          ))}
        </div>
      </motion.div>
    </div>
  );
}
