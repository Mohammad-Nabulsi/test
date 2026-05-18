import { createFileRoute } from "@tanstack/react-router";
import { motion } from "framer-motion";
import { PolarAngleAxis, PolarGrid, Radar, RadarChart, ResponsiveContainer, Tooltip } from "recharts";
import { benchmark } from "@/lib/mock-data";
import { Trophy } from "lucide-react";

export const Route = createFileRoute("/dashboard/benchmarking")({ component: Benchmarking });

function Benchmarking() {
  return (
    <div className="space-y-5">
      <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="rounded-2xl border border-border p-6 bg-mesh">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <div className="text-xs uppercase tracking-widest text-muted-foreground">ترتيبك بالقطاع</div>
            <div className="flex items-baseline gap-2 mt-1">
              <span className="num-display text-5xl font-semibold text-gradient-brand">{benchmark.percentile}</span>
              <span className="text-sm text-muted-foreground">من ١٠٠</span>
            </div>
            <p className="text-sm mt-1">إنتي متفوّقة على <strong>٨٢٪ من مشاريع الموضة</strong> بـ{benchmark.sector}.</p>
          </div>
          <div className="relative h-32 w-32">
            <svg viewBox="0 0 100 100" className="h-full w-full -rotate-90">
              <circle cx="50" cy="50" r="42" fill="none" stroke="var(--muted)" strokeWidth="8" />
              <motion.circle cx="50" cy="50" r="42" fill="none" stroke="url(#bgrad)" strokeWidth="8" strokeLinecap="round"
                strokeDasharray={`${(benchmark.percentile / 100) * 264} 264`} initial={{ strokeDasharray: "0 264" }} animate={{ strokeDasharray: `${(benchmark.percentile / 100) * 264} 264` }} transition={{ duration: 1.5, ease: "easeOut" }} />
              <defs>
                <linearGradient id="bgrad" x1="0" x2="1"><stop offset="0%" stopColor="var(--brand)" /><stop offset="100%" stopColor="var(--brand-3)" /></linearGradient>
              </defs>
            </svg>
            <div className="absolute inset-0 flex items-center justify-center"><Trophy className="h-8 w-8 text-[var(--brand)]" /></div>
          </div>
        </div>
      </motion.div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <div className="lg:col-span-2 rounded-2xl border border-border bg-card shadow-card p-5">
          <h3 className="text-sm font-semibold">مقارنة متعددة المحاور</h3>
          <p className="text-xs text-muted-foreground mb-2">إنتي مقابل متوسط القطاع</p>
          <div className="h-80">
            <ResponsiveContainer>
              <RadarChart data={benchmark.radar}>
                <PolarGrid stroke="var(--border)" />
                <PolarAngleAxis dataKey="metric" tick={{ fontSize: 11, fill: "var(--muted-foreground)" }} />
                <Tooltip contentStyle={{ background: "var(--popover)", border: "1px solid var(--border)", borderRadius: 12, fontSize: 12 }} />
                <Radar name="إنتي" dataKey="you" stroke="var(--brand)" fill="var(--brand)" fillOpacity={0.35} strokeWidth={2} />
                <Radar name="القطاع" dataKey="peer" stroke="var(--brand-3)" fill="var(--brand-3)" fillOpacity={0.18} strokeWidth={2} />
              </RadarChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="rounded-2xl border border-border bg-card shadow-card p-5">
          <h3 className="text-sm font-semibold mb-3">ترتيب القطاع</h3>
          <div className="space-y-2.5">
            {benchmark.leaderboard.map((b, i) => (
              <motion.div key={b.name} initial={{ opacity: 0, x: 8 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: i * 0.08 }} className={`flex items-center gap-3 p-2.5 rounded-xl ${b.you ? "bg-gradient-soft border border-[var(--brand)]/30" : "bg-muted/40"}`}>
                <div className="num-display text-lg font-semibold text-muted-foreground w-6">{i + 1}</div>
                <div className="flex-1 min-w-0">
                  <div className="text-sm font-medium truncate flex items-center gap-1.5">{b.name}{b.you && <span className="text-[10px] px-1.5 py-0.5 rounded bg-gradient-brand text-white">إنتي</span>}</div>
                </div>
                <div className="num-display text-sm font-semibold">{b.score}</div>
              </motion.div>
            ))}
          </div>
        </div>
      </div>

      <div className="rounded-2xl border border-border bg-card shadow-card p-5">
        <h3 className="text-sm font-semibold mb-4">تفصيل المؤشرات وحدة وحدة</h3>
        <div className="space-y-3">
          {benchmark.metrics.map((m, i) => {
            const pct = (m.you / m.top) * 100;
            const peerPct = (m.peer / m.top) * 100;
            return (
              <motion.div key={m.label} initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: i * 0.05 }}>
                <div className="flex items-center justify-between text-xs mb-1.5">
                  <span className="font-medium">{m.label}</span>
                  <span className="text-muted-foreground">إنتي <strong className="text-foreground num-display">{m.you}</strong> · القطاع {m.peer} · الأعلى {m.top}</span>
                </div>
                <div className="relative h-2.5 rounded-full bg-muted overflow-hidden">
                  <div className="absolute top-0 bottom-0 bg-muted-foreground/30" style={{ width: `${peerPct}%` }} />
                  <motion.div initial={{ width: 0 }} animate={{ width: `${pct}%` }} transition={{ duration: 1, delay: i * 0.05 }} className="absolute top-0 bottom-0 bg-gradient-brand rounded-full" />
                </div>
              </motion.div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
