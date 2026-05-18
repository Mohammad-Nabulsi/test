import { createFileRoute } from "@tanstack/react-router";
import { motion } from "framer-motion";
import { recommendations } from "@/lib/mock-data";
import { Sparkles, Clock, Hash, Type, MousePointerClick, Layout, ChevronLeft } from "lucide-react";

const icons: Record<string, any> = { "وقت النشر": Clock, "محتوى": Type, "هاشتاجات": Hash, "صيغة": Layout, "زر دعوة": MousePointerClick };

export const Route = createFileRoute("/dashboard/recommendations")({ component: Recs });

function Recs() {
  return (
    <div className="space-y-6">
      <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="relative overflow-hidden rounded-2xl border border-border p-6 bg-mesh">
        <div className="flex items-center gap-3 mb-3">
          <div className="h-10 w-10 rounded-xl bg-gradient-brand flex items-center justify-center shadow-glow"><Sparkles className="h-5 w-5 text-white" /></div>
          <div>
            <h2 className="font-display text-xl tracking-tight">٦ خطوات جاهزة تطبّقيها</h2>
            <p className="text-xs text-muted-foreground">متوقّع ترفع التفاعل ٣٨٪ خلال الـ٣٠ يوم الجاية · معدل ثقة ٨٤٪</p>
          </div>
        </div>
      </motion.div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {recommendations.map((r, i) => {
          const Icon = icons[r.category] ?? Sparkles;
          return (
            <motion.div key={r.id} initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: i * 0.06 }} whileHover={{ y: -2 }} className="group relative overflow-hidden rounded-2xl border border-border bg-card shadow-card hover:shadow-elegant transition-all p-5">
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
                  <div className="text-gradient-brand num-display text-xl font-semibold">{r.impact}</div>
                </div>
              </div>
              <div className="relative mt-4 flex items-center justify-between">
                <div className="flex items-center gap-2 flex-1 max-w-xs">
                  <span className="text-[11px] text-muted-foreground w-16">الثقة</span>
                  <div className="flex-1 h-1.5 rounded-full bg-muted overflow-hidden">
                    <motion.div initial={{ width: 0 }} animate={{ width: `${r.confidence}%` }} transition={{ duration: 1.2, delay: i * 0.05 }} className="h-full bg-gradient-brand" />
                  </div>
                  <span className="text-xs font-medium num-display">{r.confidence}%</span>
                </div>
                <button className="text-xs flex items-center gap-1 px-3 py-1.5 rounded-lg border border-border hover:bg-accent transition-colors">طبّق <ChevronLeft className="h-3 w-3" /></button>
              </div>
            </motion.div>
          );
        })}
      </div>
    </div>
  );
}
