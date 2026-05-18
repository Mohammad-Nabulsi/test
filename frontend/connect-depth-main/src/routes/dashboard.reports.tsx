import { createFileRoute } from "@tanstack/react-router";
import { motion } from "framer-motion";
import { FileText, FileBarChart, Presentation, Download } from "lucide-react";

export const Route = createFileRoute("/dashboard/reports")({ component: Reports });

const reports = [
  { icon: FileText, t: "ملخص تنفيذي", d: "صفحة وحدة بأهم إنجازات الشهر", color: "from-[var(--brand)] to-[var(--brand-2)]" },
  { icon: FileBarChart, t: "تقرير تحليلات شامل", d: "١٢ صفحة تغطّي كل المؤشرات والشرائح", color: "from-[var(--brand-2)] to-[var(--brand-3)]" },
  { icon: Presentation, t: "عرض للمستثمرين", d: "شرائح جاهزة للعرض مع قصة النمو", color: "from-[var(--brand-3)] to-[var(--brand)]" },
];

function Reports() {
  return (
    <div className="space-y-5">
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {reports.map((r, i) => (
          <motion.div key={r.t} initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: i * 0.08 }} whileHover={{ y: -3 }} className="group rounded-2xl border border-border bg-card shadow-card hover:shadow-elegant transition-all overflow-hidden">
            <div className={`relative h-40 bg-gradient-to-br ${r.color} p-5 flex items-end`}>
              <div className="absolute inset-0 bg-mesh opacity-30" />
              <div className="absolute top-4 left-4 h-9 w-9 rounded-lg bg-white/20 backdrop-blur-md flex items-center justify-center"><r.icon className="h-4 w-4 text-white" /></div>
              <div className="relative space-y-1">
                <div className="h-1.5 w-24 rounded bg-white/40" />
                <div className="h-1.5 w-16 rounded bg-white/30" />
                <div className="h-1.5 w-20 rounded bg-white/30" />
              </div>
            </div>
            <div className="p-5">
              <div className="font-display text-base">{r.t}</div>
              <p className="text-xs text-muted-foreground mt-1">{r.d}</p>
              <button className="mt-4 w-full h-9 rounded-lg bg-gradient-brand text-white text-xs font-medium flex items-center justify-center gap-2 shadow-glow hover:opacity-90 transition-opacity"><Download className="h-3.5 w-3.5" />صدّر PDF</button>
            </div>
          </motion.div>
        ))}
      </div>

      <div className="rounded-2xl border border-border bg-card shadow-card p-5">
        <h3 className="text-sm font-semibold mb-3">آخر التقارير</h3>
        <div className="divide-y divide-border">
          {[
            { n: "أداء أكتوبر ٢٠٢٥", d: "PDF · ٢.٤ ميجا", t: "قبل يومين" },
            { n: "تقرير حملة العيد", d: "PPTX · ٥.١ ميجا", t: "قبل أسبوع" },
            { n: "ملخص الربع الثالث", d: "PDF · ١.١ ميجا", t: "قبل ٣ أسابيع" },
          ].map((x, i) => (
            <div key={i} className="flex items-center justify-between py-3">
              <div className="flex items-center gap-3">
                <div className="h-9 w-9 rounded-lg bg-accent flex items-center justify-center"><FileText className="h-4 w-4 text-[var(--brand)]" /></div>
                <div>
                  <div className="text-sm font-medium">{x.n}</div>
                  <div className="text-[11px] text-muted-foreground">{x.d} · {x.t}</div>
                </div>
              </div>
              <button className="text-xs px-3 py-1.5 rounded-lg border border-border hover:bg-accent">تنزيل</button>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
