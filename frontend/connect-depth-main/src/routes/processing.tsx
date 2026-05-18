import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { motion, AnimatePresence } from "framer-motion";
import { useEffect, useState } from "react";
import { Check, Loader2, Sparkles, Zap } from "lucide-react";
import { insightStream } from "@/lib/mock-data";
import { consumePendingDatasetFile, saveLatestCaptionAnalysis } from "@/lib/upload-state";

export const Route = createFileRoute("/processing")({ component: Processing });
const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000";

const stages = [
  "تحميل البيانات",
  "تنظيف وتنسيق",
  "استخراج المؤشرات",
  "اكتشاف الهاشتاجات",
  "تحليل اللغة الطبيعية",
  "تحليل المشاعر",
  "تجميع البوستات",
  "توليد التوصيات",
  "توقع الترندات",
  "تجهيز لوحتك",
];

function Processing() {
  const navigate = useNavigate();
  const [stage, setStage] = useState(0);
  const [progress, setProgress] = useState(0);
  const [insights, setInsights] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [realRun, setRealRun] = useState(false);

  useEffect(() => {
    const file = consumePendingDatasetFile();
    if (file) {
      let cancelled = false;
      setRealRun(true);
      setInsights((prev) => ["تم استلام الملف، جاري تشغيل preprocess ثم KPI ثم مرحلة الهاشتاقات بالتوازي.", ...prev].slice(0, 4));

      const startedAt = Date.now();
      const progressTimer = setInterval(() => {
        const elapsed = Date.now() - startedAt;
        const p = Math.min(92, 8 + (elapsed / 90000) * 84);
        setProgress(p);
        setStage(Math.min(stages.length - 2, Math.floor((p / 100) * stages.length)));
      }, 500);

      const form = new FormData();
      form.append("file", file);
      fetch(`${API_BASE}/api/datasets/upload-preprocess-kpi`, { method: "POST", body: form })
        .then(async (response) => {
          if (!response.ok) throw new Error(await response.text());
          return response.json();
        })
        .then((result) => {
          if (cancelled) return;
          saveLatestCaptionAnalysis({
            dataset_id: String(result.dataset_id || ""),
            rows_received: Number(result.rows_received || 0),
            rows_cleaned: Number(result.rows_cleaned || 0),
            rows_kpi: Number(result.rows_kpi || 0),
          });
          setProgress(100);
          setStage(stages.length - 1);
          setInsights((prev) => ["اكتمل التحليل. سيتم فتح صفحة تجميع المحتوى بالنتائج.", ...prev].slice(0, 4));
          setTimeout(() => navigate({ to: "/dashboard/hashtags" }), 700);
        })
        .catch((err) => {
          if (cancelled) return;
          setError(err instanceof Error ? err.message : "Upload pipeline failed.");
        })
        .finally(() => clearInterval(progressTimer));

      return () => {
        cancelled = true;
        clearInterval(progressTimer);
      };
    }

    const total = 6500;
    const t0 = Date.now();
    const tick = setInterval(() => {
      const e = Date.now() - t0;
      const p = Math.min(100, (e / total) * 100);
      setProgress(p);
      setStage(Math.min(stages.length - 1, Math.floor((p / 100) * stages.length)));
      if (p >= 100) {
        clearInterval(tick);
        setTimeout(() => navigate({ to: "/dashboard" }), 600);
      }
    }, 80);
    return () => clearInterval(tick);
  }, [navigate]);

  useEffect(() => {
    let i = 0;
    const t = setInterval(() => {
      setInsights((prev) => [insightStream[i % insightStream.length], ...prev].slice(0, 4));
      i++;
    }, 900);
    return () => clearInterval(t);
  }, []);

  return (
    <div className="min-h-screen bg-background relative overflow-hidden flex items-center justify-center p-4" dir="rtl">
      <div className="absolute inset-0 bg-mesh" />
      <div className="absolute inset-0 grid-bg opacity-30" />

      {Array.from({ length: 14 }).map((_, i) => (
        <motion.div
          key={i}
          className="absolute h-1.5 w-1.5 rounded-full bg-gradient-brand"
          style={{ left: `${(i * 73) % 100}%`, top: `${(i * 37) % 100}%` }}
          animate={{ y: [0, -40, 0], opacity: [0.2, 0.8, 0.2] }}
          transition={{ duration: 4 + (i % 3), repeat: Infinity, delay: i * 0.2 }}
        />
      ))}

      <div className="relative max-w-5xl w-full grid grid-cols-1 lg:grid-cols-5 gap-6">
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="lg:col-span-3 rounded-3xl border border-border bg-card/80 backdrop-blur-xl shadow-elegant p-6 md:p-8">
          <div className="flex items-center gap-3 mb-2">
            <motion.div animate={{ rotate: 360 }} transition={{ duration: 4, repeat: Infinity, ease: "linear" }} className="h-10 w-10 rounded-xl bg-gradient-brand flex items-center justify-center shadow-glow">
              <Zap className="h-5 w-5 text-white" />
            </motion.div>
            <div>
              <div className="font-display text-xl tracking-tight">نحن نجهز ذكاء عملك</div>
              <div className="text-xs text-muted-foreground">بوتيك ليالي · ٣١٢ بوست · إنستغرام + تيك توك</div>
            </div>
          </div>

          <div className="mt-5">
            <div className="flex items-baseline justify-between mb-1.5">
              <span className="text-xs text-muted-foreground">التقدم العام</span>
              <span className="num-display text-2xl font-semibold text-gradient-brand">{Math.round(progress)}%</span>
            </div>
            <div className="h-2 rounded-full bg-muted overflow-hidden">
              <motion.div className="h-full bg-gradient-brand rounded-full" animate={{ width: `${progress}%` }} transition={{ duration: 0.2 }} />
            </div>
          </div>

          <div className="mt-6 space-y-2">
            {stages.map((s, i) => {
              const done = i < stage;
              const active = i === stage;
              return (
                <motion.div key={s} initial={{ opacity: 0, x: 8 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: i * 0.04 }} className={`flex items-center gap-3 p-2.5 rounded-xl transition-all ${active ? "bg-accent/60 border border-[var(--brand)]/30" : "bg-transparent"}`}>
                  <div className={`h-7 w-7 rounded-lg flex items-center justify-center shrink-0 ${done ? "bg-success text-white" : active ? "bg-gradient-brand text-white" : "bg-muted text-muted-foreground"}`}>
                    {done ? <Check className="h-4 w-4" /> : active ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <span className="text-[10px]">{i + 1}</span>}
                  </div>
                  <span className={`text-sm flex-1 ${active ? "font-medium" : done ? "text-muted-foreground line-through" : "text-muted-foreground"}`}>{s}</span>
                  {active && <span className="text-[10px] text-[var(--brand)] uppercase tracking-widest">جاري العمل...</span>}
                </motion.div>
              );
            })}
          </div>
        </motion.div>

        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }} className="lg:col-span-2 space-y-4">
          <div className="rounded-3xl border border-border bg-card/80 backdrop-blur-xl shadow-elegant p-5">
            <div className="text-xs uppercase tracking-widest text-muted-foreground">اكتشاف مباشر</div>
            {realRun && (
              <div className="mt-3 rounded-xl border border-[var(--brand)]/20 bg-accent/40 p-3 text-xs text-muted-foreground">
                تشغيل فعلي: upload ثم preprocess ثم KPI
              </div>
            )}
            {error && (
              <div className="mt-3 rounded-xl border border-destructive/30 bg-destructive/5 p-3 text-xs text-destructive">
                فشل التحليل: {error}
              </div>
            )}
            <div className="grid grid-cols-2 gap-3 mt-3">
              {[
                { l: "بوستات تمت قراءتها", v: Math.round((progress / 100) * 312) },
                { l: "هاشتاجات مكتشفة", v: Math.round((progress / 100) * 184) },
                { l: "مواضيع مرصودة", v: Math.round((progress / 100) * 12) },
                { l: "حالات شاذة", v: Math.round((progress / 100) * 9) },
              ].map((x) => (
                <div key={x.l} className="rounded-xl bg-muted/40 p-3">
                  <div className="text-[10px] text-muted-foreground">{x.l}</div>
                  <div className="num-display text-2xl font-semibold mt-0.5">{x.v}</div>
                </div>
              ))}
            </div>
          </div>

          <div className="rounded-3xl border border-border bg-card/80 backdrop-blur-xl shadow-elegant p-5">
            <div className="flex items-center gap-2 mb-3">
              <Sparkles className="h-4 w-4 text-[var(--brand)]" />
              <div className="text-xs uppercase tracking-widest text-muted-foreground">رؤى نكتشفها الآن</div>
            </div>
            <div className="space-y-2 min-h-[160px]">
              <AnimatePresence>
                {insights.map((t, i) => (
                  <motion.div key={t + i} initial={{ opacity: 0, x: -12, height: 0 }} animate={{ opacity: 1, x: 0, height: "auto" }} exit={{ opacity: 0 }} className="text-xs p-3 rounded-xl bg-gradient-soft border border-border/60 leading-relaxed">
                    {t}
                  </motion.div>
                ))}
              </AnimatePresence>
            </div>
          </div>
        </motion.div>
      </div>
    </div>
  );
}
