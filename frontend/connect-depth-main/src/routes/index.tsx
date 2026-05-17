import { createFileRoute, Link, useNavigate } from "@tanstack/react-router";
import { motion } from "framer-motion";
import { useState, useRef } from "react";
import { Upload, FileSpreadsheet, Sparkles, BarChart3, Layers, Target, TrendingUp, Brain, ArrowLeft, Zap, Instagram, Facebook, Music2 } from "lucide-react";
import { setPendingDatasetFile } from "@/lib/upload-state";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "سهيل — ذكاء وسائل التواصل للمشاريع الفلسطينية" },
      { name: "description", content: "حمّل بيانات إنستغرام أو فيسبوك أو تيك توك، واحصل على مؤشرات أداء وتوصيات وتحليلات بالذكاء الاصطناعي بأقل من دقيقة." },
      { property: "og:title", content: "سهيل — ذكاء وسائل التواصل" },
      { property: "og:description", content: "تحليلات وذكاء اصطناعي مصمّم للمشاريع الصغيرة والمتوسطة في فلسطين والشام." },
    ],
  }),
  component: Landing,
});

const features = [
  { icon: BarChart3, t: "تحليل المؤشرات", d: "تفاعل، نمو، وصول — بنظرة وحدة" },
  { icon: Sparkles, t: "توصيات ذكية", d: "خطوات مرتّبة حسب الأثر المتوقّع" },
  { icon: TrendingUp, t: "توقع الترندات", d: "توقعات ٣٠ يوم مع نطاقات ثقة" },
  { icon: Layers, t: "تجميع المحتوى", d: "اكتشف المواضيع اللي بتشتغل" },
  { icon: Target, t: "مقارنة بالقطاع", d: "شوف وين ترتيبك بين أمثالك" },
  { icon: Brain, t: "رؤى بلغة طبيعية", d: "إجابات واضحة، عربي أو إنجليزي" },
];

function Landing() {
  const navigate = useNavigate();
  const [drag, setDrag] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);
  const startUpload = (file?: File) => {
    if (!file) return;
    setPendingDatasetFile(file);
    navigate({ to: "/processing" });
  };

  return (
    <div className="min-h-screen bg-background">
      {/* القائمة */}
      <nav className="sticky top-0 z-30 glass border-b border-border/60">
        <div className="max-w-7xl mx-auto px-4 md:px-6 h-16 flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <div className="h-9 w-9 rounded-xl bg-gradient-brand flex items-center justify-center shadow-glow"><Zap className="h-4 w-4 text-white" strokeWidth={2.5} /></div>
            <div>
              <div className="font-display font-semibold tracking-tight">سهيل</div>
              <div className="text-[10px] uppercase tracking-widest text-muted-foreground -mt-0.5">ذكاء وسائل التواصل</div>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <a href="#features" className="text-sm text-muted-foreground hover:text-foreground hidden md:inline">المزايا</a>
            <a href="#upload" className="text-sm text-muted-foreground hover:text-foreground hidden md:inline">رفع البيانات</a>
            <Link to="/dashboard" className="text-xs md:text-sm h-9 px-4 rounded-lg bg-gradient-brand text-white font-medium flex items-center gap-1.5 shadow-glow hover:opacity-90 transition">افتح اللوحة <ArrowLeft className="h-3.5 w-3.5" /></Link>
          </div>
        </div>
      </nav>

      {/* الهيرو */}
      <section className="relative overflow-hidden">
        <div className="absolute inset-0 bg-mesh" />
        <div className="absolute inset-0 grid-bg opacity-40" />
        <div className="relative max-w-7xl mx-auto px-4 md:px-6 pt-20 pb-16 md:pt-28 md:pb-24">
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.6 }} className="max-w-3xl">
            <div className="inline-flex items-center gap-2 text-xs font-medium px-3 py-1.5 rounded-full bg-card/80 border border-border backdrop-blur-md">
              <span className="h-1.5 w-1.5 rounded-full bg-success animate-pulse" />
              مصمّم للمشاريع الفلسطينية والشامية
            </div>
            <h1 className="mt-5 font-display text-5xl md:text-7xl tracking-tight leading-[1.05]">
              حوّل بيانات السوشال ميديا تبعك لـ<span className="text-gradient-brand">قرارات</span>.
            </h1>
            <p className="mt-5 text-lg text-muted-foreground max-w-2xl">
              حمّل بيانات إنستغرام أو فيسبوك أو تيك توك. بأقل من دقيقة، سهيل بيطلّعلك المؤشرات، بيجمّع محتواك، بيقارنك مع قطاعك، وبيحكيلك بالظبط شو لازم تنشر بعدها.
            </p>
            <div className="mt-7 flex flex-wrap items-center gap-3">
              <Link to="/processing" className="h-12 px-6 rounded-xl bg-gradient-brand text-white font-medium flex items-center gap-2 shadow-glow hover:opacity-90 transition">
                <Sparkles className="h-4 w-4" /> شغّل التحليل التجريبي
              </Link>
              <a href="#upload" className="h-12 px-6 rounded-xl border border-border bg-card font-medium flex items-center gap-2 hover:bg-accent transition">حمّل بياناتك إنت</a>
            </div>
            <div className="mt-6 flex items-center gap-4 text-xs text-muted-foreground">
              <span>مدعوم:</span>
              <span className="flex items-center gap-1.5"><Instagram className="h-3.5 w-3.5" /> إنستغرام</span>
              <span className="flex items-center gap-1.5"><Facebook className="h-3.5 w-3.5" /> فيسبوك</span>
              <span className="flex items-center gap-1.5"><Music2 className="h-3.5 w-3.5" /> تيك توك</span>
            </div>
          </motion.div>

          {/* معاينة طافية */}
          <motion.div initial={{ opacity: 0, y: 30 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.3, duration: 0.7 }} className="mt-14 relative">
            <div className="relative rounded-3xl border border-border bg-card shadow-elegant overflow-hidden">
              <div className="h-9 bg-muted/60 border-b border-border flex items-center px-4 gap-1.5">
                <span className="h-2.5 w-2.5 rounded-full bg-destructive/60" />
                <span className="h-2.5 w-2.5 rounded-full bg-warning/60" />
                <span className="h-2.5 w-2.5 rounded-full bg-success/60" />
                <span className="mr-3 text-[11px] text-muted-foreground" dir="ltr">suhail.app/dashboard</span>
              </div>
              <div className="grid grid-cols-4 gap-3 p-4 bg-mesh">
                {[
                  { l: "التفاعل", v: "7.82%", d: "+14%" },
                  { l: "متوسط اللايكات", v: "2,143", d: "+9%" },
                  { l: "متوسط المشاهدات", v: "18.4k", d: "+38%" },
                  { l: "النمو", v: "12.4%", d: "+4%" },
                ].map((x, i) => (
                  <motion.div key={i} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.6 + i * 0.1 }} className="rounded-xl bg-card border border-border p-3">
                    <div className="text-[10px] text-muted-foreground">{x.l}</div>
                    <div className="num-display text-xl font-semibold mt-1">{x.v}</div>
                    <div className="text-[10px] text-success">{x.d}</div>
                  </motion.div>
                ))}
              </div>
            </div>
            <motion.div animate={{ y: [0, -10, 0] }} transition={{ duration: 4, repeat: Infinity }} className="absolute -top-6 -left-4 md:left-12 w-56 rounded-xl border border-border bg-card shadow-elegant p-3 hidden md:block">
              <div className="flex items-center gap-2"><div className="h-7 w-7 rounded-lg bg-gradient-brand flex items-center justify-center"><Sparkles className="h-3.5 w-3.5 text-white" /></div><div className="text-[11px] font-medium">رؤية ذكية</div></div>
              <p className="text-[11px] text-muted-foreground mt-2">الريلز اللي بعد الساعة ٧ مساءً أداؤها أحسن بـ<strong className="text-foreground">٤٢٪</strong>.</p>
            </motion.div>
            <motion.div animate={{ y: [0, 8, 0] }} transition={{ duration: 5, repeat: Infinity, delay: 0.5 }} className="absolute -bottom-6 -right-2 md:right-12 w-52 rounded-xl border border-border bg-card shadow-elegant p-3 hidden md:block">
              <div className="text-[10px] text-muted-foreground">مجموعة مكتشفة</div>
              <div className="text-sm font-medium mt-1">عروض فاخرة</div>
              <div className="text-[11px] text-success mt-0.5">٤٢ بوست · ٩.٢٪ تفاعل</div>
            </motion.div>
          </motion.div>
        </div>
      </section>

      {/* رفع البيانات */}
      <section id="upload" className="max-w-5xl mx-auto px-4 md:px-6 py-16">
        <div className="text-center mb-8">
          <h2 className="font-display text-3xl md:text-4xl tracking-tight">حمّل بياناتك، خد رؤى.</h2>
          <p className="text-muted-foreground mt-2">CSV، إكسل أو JSON · لحدّ ٥٠ ميجا</p>
        </div>
        <motion.div
          onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
          onDragLeave={() => setDrag(false)}
          onDrop={(e) => { e.preventDefault(); setDrag(false); startUpload(e.dataTransfer.files[0]); }}
          onClick={() => fileRef.current?.click()}
          whileHover={{ scale: 1.005 }}
          className={`relative cursor-pointer rounded-3xl border-2 border-dashed p-12 md:p-16 text-center transition-all ${drag ? "border-[var(--brand)] bg-accent/40 shadow-glow" : "border-border bg-card hover:border-[var(--brand)]/40"}`}
        >
          <input ref={fileRef} type="file" className="hidden" accept=".csv,.json" onChange={(e) => startUpload(e.target.files?.[0])} />
          <div className="absolute inset-0 bg-mesh opacity-40 rounded-3xl pointer-events-none" />
          <div className="relative">
            <motion.div animate={{ y: [0, -8, 0] }} transition={{ duration: 3, repeat: Infinity }} className="mx-auto h-16 w-16 rounded-2xl bg-gradient-brand flex items-center justify-center shadow-glow">
              <Upload className="h-7 w-7 text-white" />
            </motion.div>
            <div className="mt-5 font-display text-xl">اسحب وافلت ملف بياناتك هون</div>
            <p className="text-sm text-muted-foreground mt-1.5">أو <span className="text-[var(--brand)] font-medium">اضغط لاختيار ملف</span> · CSV، XLSX، JSON</p>
            <div className="mt-6 flex items-center justify-center gap-2 flex-wrap">
              <span className="px-3 py-1.5 rounded-full text-xs bg-card border border-border flex items-center gap-1.5"><FileSpreadsheet className="h-3 w-3" /> instagram-export.csv</span>
              <span className="px-3 py-1.5 rounded-full text-xs bg-card border border-border flex items-center gap-1.5"><FileSpreadsheet className="h-3 w-3" /> tiktok-data.json</span>
            </div>
            <div className="mt-8">
              <Link to="/processing" className="inline-flex items-center gap-2 h-11 px-6 rounded-xl bg-gradient-brand text-white font-medium shadow-glow hover:opacity-90 transition" onClick={(e) => e.stopPropagation()}>
                <Sparkles className="h-4 w-4" /> جرّب البيانات التجريبية
              </Link>
            </div>
          </div>
        </motion.div>

        {/* مشاريع حديثة */}
        <div className="mt-10">
          <h3 className="text-sm font-semibold mb-3 text-muted-foreground uppercase tracking-widest">مشاريع حديثة</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
            {[
              { n: "بوتيك ليالي", t: "أزياء · ٣١٢ بوست", d: "اليوم", c: "from-fuchsia-500 to-violet-600" },
              { n: "بيت القهوة", t: "مأكولات · ١٨٧ بوست", d: "أمس", c: "from-amber-500 to-orange-600" },
              { n: "حِرف نابلس", t: "حرف يدوية · ٩٤ بوست", d: "قبل ٣ أيام", c: "from-emerald-500 to-teal-600" },
            ].map((p) => (
              <Link key={p.n} to="/dashboard" className="group rounded-2xl border border-border bg-card p-4 flex items-center gap-3 hover:shadow-elegant transition-all">
                <div className={`h-11 w-11 rounded-xl bg-gradient-to-br ${p.c}`} />
                <div className="flex-1 min-w-0">
                  <div className="text-sm font-medium truncate">{p.n}</div>
                  <div className="text-[11px] text-muted-foreground truncate">{p.t}</div>
                </div>
                <div className="text-[11px] text-muted-foreground">{p.d}</div>
              </Link>
            ))}
          </div>
        </div>
      </section>

      {/* المزايا */}
      <section id="features" className="max-w-7xl mx-auto px-4 md:px-6 py-16">
        <div className="text-center mb-10">
          <h2 className="font-display text-3xl md:text-4xl tracking-tight">كل اللي بحتاجه صاحب مشروع.</h2>
          <p className="text-muted-foreground mt-2">بدون ما تحتاج فريق بيانات.</p>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {features.map((f, i) => (
            <motion.div key={f.t} initial={{ opacity: 0, y: 12 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }} transition={{ delay: i * 0.05 }} className="group rounded-2xl border border-border bg-card p-5 hover:shadow-elegant transition-all">
              <div className="h-10 w-10 rounded-xl bg-accent flex items-center justify-center group-hover:bg-gradient-brand group-hover:shadow-glow transition-all">
                <f.icon className="h-5 w-5 text-[var(--brand)] group-hover:text-white transition-colors" />
              </div>
              <div className="mt-3 font-display text-lg">{f.t}</div>
              <p className="text-sm text-muted-foreground mt-1">{f.d}</p>
            </motion.div>
          ))}
        </div>
      </section>

      <footer className="border-t border-border py-8 text-center text-xs text-muted-foreground">
        مصنوع بحب للمشاريع الصغيرة والمتوسطة في فلسطين والشام · سهيل
      </footer>
    </div>
  );
}
