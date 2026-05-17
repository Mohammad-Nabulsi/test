import { motion, useMotionValue, useTransform, animate } from "framer-motion";
import { ArrowDownRight, ArrowUpRight } from "lucide-react";
import { useEffect } from "react";
import { Area, AreaChart, ResponsiveContainer } from "recharts";

type Props = {
  label: string;
  value: number | string;
  suffix?: string;
  delta: number;
  spark: number[];
  string?: boolean;
  index?: number;
};

function Counter({ to }: { to: number }) {
  const mv = useMotionValue(0);
  const rounded = useTransform(mv, (v) => (to >= 100 ? Math.round(v).toLocaleString() : v.toFixed(2)));
  useEffect(() => { const c = animate(mv, to, { duration: 1.2, ease: "easeOut" }); return c.stop; }, [to, mv]);
  return <motion.span>{rounded}</motion.span>;
}

export function KpiCard({ label, value, suffix, delta, spark, string, index = 0 }: Props) {
  const positive = delta >= 0;
  const data = spark.map((v, i) => ({ i, v }));
  return (
    <motion.div
      initial={{ opacity: 0, y: 14 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.05, duration: 0.4, ease: "easeOut" }}
      whileHover={{ y: -2 }}
      className="group relative overflow-hidden rounded-2xl border border-border bg-card shadow-card p-5 hover:shadow-elegant transition-shadow"
    >
      <div className="absolute -top-12 -right-12 h-32 w-32 rounded-full bg-gradient-brand opacity-0 group-hover:opacity-10 blur-2xl transition-opacity" />
      <div className="flex items-start justify-between">
        <div className="text-xs font-medium text-muted-foreground">{label}</div>
        {!!delta && (
          <div className={`flex items-center gap-0.5 text-[11px] font-medium px-1.5 py-0.5 rounded-md ${positive ? "text-success bg-success/10" : "text-destructive bg-destructive/10"}`}>
            {positive ? <ArrowUpRight className="h-3 w-3" /> : <ArrowDownRight className="h-3 w-3" />}
            {Math.abs(delta)}%
          </div>
        )}
      </div>
      <div className="mt-3 num-display text-3xl font-semibold tracking-tight">
        {string ? value : <Counter to={typeof value === "number" ? value : 0} />}
        {suffix && <span className="text-lg text-muted-foreground ml-0.5">{suffix}</span>}
      </div>
      <div className="mt-3 h-10 -mx-1">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={data}>
            <defs>
              <linearGradient id={`spark-${label}`} x1="0" x2="0" y1="0" y2="1">
                <stop offset="0%" stopColor="var(--brand)" stopOpacity={0.5} />
                <stop offset="100%" stopColor="var(--brand)" stopOpacity={0} />
              </linearGradient>
            </defs>
            <Area type="monotone" dataKey="v" stroke="var(--brand)" strokeWidth={1.8} fill={`url(#spark-${label})`} />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </motion.div>
  );
}
