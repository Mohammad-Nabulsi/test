import { createFileRoute } from "@tanstack/react-router";
import { motion } from "framer-motion";
import { Area, ComposedChart, Line, ResponsiveContainer, Tooltip, XAxis, YAxis, CartesianGrid, ReferenceLine } from "recharts";
import { forecast } from "@/lib/mock-data";

export const Route = createFileRoute("/dashboard/forecasting")({ component: Forecasting });

function Forecasting() {
  return (
    <div className="space-y-5">
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {[
          { l: "التفاعل المتوقع (٣٠ يوم)", v: "+34%", s: "مقارنة بآخر ٣٠ يوم", c: "from-[var(--brand)] to-[var(--brand-2)]" },
          { l: "المتابعون المتوقعون (٣٠ يوم)", v: "54,820", s: "نمو ١٣.٧٪", c: "from-[var(--brand-2)] to-[var(--brand-3)]" },
          { l: "ثقة التوقع", v: "91%", s: "ARIMA + Prophet مدمج", c: "from-[var(--brand-3)] to-[var(--brand)]" },
        ].map((x, i) => (
          <motion.div key={i} initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: i * 0.06 }} className="relative overflow-hidden rounded-2xl border border-border bg-card shadow-card p-5">
            <div className={`absolute -top-10 -left-10 h-32 w-32 rounded-full bg-gradient-to-br ${x.c} opacity-15 blur-2xl`} />
            <div className="text-xs text-muted-foreground">{x.l}</div>
            <div className="num-display text-3xl font-semibold mt-1">{x.v}</div>
            <div className="text-[11px] text-muted-foreground mt-0.5">{x.s}</div>
          </motion.div>
        ))}
      </div>

      <div className="rounded-2xl border border-border bg-card shadow-card p-5">
        <div className="flex items-center justify-between mb-3">
          <div>
            <h3 className="text-sm font-semibold">توقع التفاعل · أفق ٦٠ يوم</h3>
            <p className="text-xs text-muted-foreground">خط متّصل = فعلي · متقطّع = متوقع · النطاق = ثقة ٩٥٪</p>
          </div>
        </div>
        <div className="h-96">
          <ResponsiveContainer>
            <ComposedChart data={forecast} margin={{ left: -12 }}>
              <defs>
                <linearGradient id="band" x1="0" x2="0" y1="0" y2="1"><stop offset="0%" stopColor="var(--brand)" stopOpacity={0.25} /><stop offset="100%" stopColor="var(--brand)" stopOpacity={0} /></linearGradient>
              </defs>
              <CartesianGrid stroke="var(--border)" strokeDasharray="3 3" vertical={false} />
              <XAxis dataKey="day" tick={{ fontSize: 10, fill: "var(--muted-foreground)" }} tickLine={false} axisLine={false} interval={6} reversed />
              <YAxis tick={{ fontSize: 11, fill: "var(--muted-foreground)" }} tickLine={false} axisLine={false} orientation="right" />
              <Tooltip contentStyle={{ background: "var(--popover)", border: "1px solid var(--border)", borderRadius: 12, fontSize: 12 }} />
              <ReferenceLine x="يوم 30" stroke="var(--border)" strokeDasharray="4 4" label={{ value: "اليوم", fill: "var(--muted-foreground)", fontSize: 10, position: "top" }} />
              <Area type="monotone" dataKey="upper" stroke="none" fill="url(#band)" />
              <Area type="monotone" dataKey="lower" stroke="none" fill="var(--background)" />
              <Line type="monotone" dataKey="actual" stroke="var(--brand)" strokeWidth={2.4} dot={false} />
              <Line type="monotone" dataKey="predicted" stroke="var(--brand)" strokeWidth={2.4} strokeDasharray="6 4" dot={false} />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="rounded-2xl border border-border bg-card shadow-card p-5">
          <h3 className="text-sm font-semibold mb-2">الأنماط الموسمية المكتشفة</h3>
          <ul className="text-sm space-y-2 text-muted-foreground">
            <li>• القمة الأسبوعية: <strong className="text-foreground">الجمعة ٨–١٠ مساءً</strong></li>
            <li>• الدورة الشهرية: نزول بالأسبوع الأول، انتعاش بالأسبوع الثالث</li>
            <li>• ارتفاع الأعياد: <strong className="text-foreground">+٣٨٪</strong> حوالين العيد</li>
          </ul>
        </div>
        <div className="rounded-2xl border border-border bg-gradient-soft p-5">
          <h3 className="text-sm font-semibold mb-2">إسقاط الترند</h3>
          <p className="text-sm">إذا ضليتي على نفس الوتيرة، رح توصلي <strong className="text-gradient-brand">٦٢ ألف متابع بحلول الربع الثاني</strong> — زيادة ٢٨٪ فوق الخط الأساسي.</p>
        </div>
      </div>
    </div>
  );
}
