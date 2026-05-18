import { createFileRoute } from "@tanstack/react-router";
import { motion } from "framer-motion";
import { useEffect, useMemo, useState } from "react";
import { Area, AreaChart, Bar, BarChart, CartesianGrid, Cell, Pie, PieChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { KpiCard } from "@/components/dashboard/KpiCard";
import { business as fallbackBusiness, kpis as fallbackKpis, engagementOverTime, heatmapData, contentTypeData, topHashtags, posts, insightStream } from "@/lib/mock-data";
import { Sparkles, TrendingUp } from "lucide-react";
import { fetchMeData, formatCompactArabic, resolveDatasetId, type MeBusiness, type MeKpi } from "@/lib/me-api";

export const Route = createFileRoute("/dashboard/")({ component: Overview });

function Overview() {
  const [hydrated, setHydrated] = useState(false);
  const [meBusiness, setMeBusiness] = useState<MeBusiness>(fallbackBusiness);
  const [meKpis, setMeKpis] = useState<MeKpi[]>(fallbackKpis as MeKpi[]);

  useEffect(() => {
    setHydrated(true);
  }, []);

  useEffect(() => {
    const datasetId = resolveDatasetId();
    fetchMeData(datasetId)
      .then((data) => {
        setMeBusiness(data.business);
        setMeKpis(data.kpis);
      })
      .catch(() => {
        setMeBusiness(fallbackBusiness);
        setMeKpis(fallbackKpis as MeKpi[]);
      });
  }, []);

  const kpiByKey = useMemo(
    () => Object.fromEntries(meKpis.map((k) => [k.key, k])) as Record<string, MeKpi>,
    [meKpis],
  );
  const visibleKpis = useMemo(
    () =>
      ["engagement", "likes", "comments", "views"]
        .map((key) => kpiByKey[key])
        .filter((k): k is MeKpi => Boolean(k)),
    [kpiByKey],
  );
  const safeDays = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
  const rawDays = useMemo(() => Array.from(new Set(heatmapData.map((d) => d.day))), []);
  const safeHeatmapData = useMemo(
    () =>
      heatmapData.map((d) => {
        const idx = rawDays.indexOf(d.day);
        return { ...d, day: idx >= 0 ? safeDays[idx] ?? d.day : d.day };
      }),
    [rawDays],
  );
  const safeEngagementOverTime = useMemo(
    () => engagementOverTime.map((p, i) => ({ ...p, date: `Day ${i + 1}` })),
    [],
  );
  const safeContentTypeData = useMemo(() => {
    const labels = ["Reels", "Carousel", "Images", "Stories"];
    return contentTypeData.map((d, i) => ({ ...d, name: labels[i] ?? d.name }));
  }, []);
  const safeTopHashtags = useMemo(
    () => topHashtags.map((h, i) => ({ ...h, tag: `#tag${i + 1}` })),
    [],
  );
  const safeInsightStream = useMemo(
    () => insightStream.map((_, i) => `Actionable insight ${i + 1} for content performance.`),
    [],
  );
  const safePosts = useMemo(
    () =>
      posts.map((p, i) => ({
        ...p,
        caption: `Sample post ${i + 1}`,
        typeLabel: p.type === "reel" ? "Reel" : p.type === "carousel" ? "Carousel" : "Image",
      })),
    [],
  );

  if (!hydrated) return null;

  return (
    <div className="space-y-6">
      {/* Hero */}
      <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="relative overflow-hidden rounded-2xl border border-border p-5 md:p-6 bg-mesh">
        <div className="relative flex flex-col md:flex-row md:items-center gap-4 justify-between">
          <div>
            <div className="inline-flex items-center gap-1.5 text-xs font-medium px-2.5 py-1 rounded-full bg-card/80 border border-border backdrop-blur-md">
              <span className="h-1.5 w-1.5 rounded-full bg-success animate-pulse" /> Live · Updated recently
            </div>
            <h2 className="mt-3 font-display text-2xl md:text-3xl tracking-tight">
              You are outperforming <span className="text-gradient-brand">82% of fashion accounts</span>.
            </h2>
            <p className="text-sm text-muted-foreground mt-1.5 max-w-2xl">Engagement improved this week, driven by evening reels and stronger content mix.</p>
          </div>
          <div className="flex gap-2">
            <div className="rounded-xl bg-card/80 backdrop-blur-md border border-border px-4 py-3 text-center">
              <div className="num-display text-2xl font-semibold">{meBusiness.posts.toLocaleString("ar-EG")}</div>
              <div className="text-[11px] text-muted-foreground uppercase tracking-wider">Posts analyzed</div>
            </div>
            <div className="rounded-xl bg-card/80 backdrop-blur-md border border-border px-4 py-3 text-center">
              <div className="num-display text-2xl font-semibold">{formatCompactArabic(meBusiness.followers)}</div>
              <div className="text-[11px] text-muted-foreground uppercase tracking-wider">Followers</div>
            </div>
          </div>
        </div>
      </motion.div>

      {/* KPI Grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {visibleKpis.map((k, i) => (
          <KpiCard key={k.key} index={i} label={k.label} value={k.value} suffix={k.suffix} delta={k.delta} spark={k.spark} string={k.string} />
        ))}
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <div className="lg:col-span-2 rounded-2xl border border-border bg-card shadow-card p-5">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h3 className="text-sm font-semibold">Engagement Over Time</h3>
              <p className="text-xs text-muted-foreground">Last 30 days · reach and engagement</p>
            </div>
            <div className="flex items-center gap-3 text-xs">
              <span className="flex items-center gap-1.5"><span className="h-2 w-2 rounded-full bg-[var(--brand)]" />Engagement</span>
              <span className="flex items-center gap-1.5"><span className="h-2 w-2 rounded-full bg-[var(--brand-3)]" />Reach</span>
            </div>
          </div>
          <div className="h-72">
            <ResponsiveContainer>
              <AreaChart data={safeEngagementOverTime} margin={{ left: -16, right: 8, top: 4 }}>
                <defs>
                  <linearGradient id="g1" x1="0" x2="0" y1="0" y2="1"><stop offset="0%" stopColor="var(--brand)" stopOpacity={0.4} /><stop offset="100%" stopColor="var(--brand)" stopOpacity={0} /></linearGradient>
                  <linearGradient id="g2" x1="0" x2="0" y1="0" y2="1"><stop offset="0%" stopColor="var(--brand-3)" stopOpacity={0.3} /><stop offset="100%" stopColor="var(--brand-3)" stopOpacity={0} /></linearGradient>
                </defs>
                <CartesianGrid stroke="var(--border)" strokeDasharray="3 3" vertical={false} />
                <XAxis dataKey="date" tick={{ fontSize: 11, fill: "var(--muted-foreground)" }} tickLine={false} axisLine={false} interval={4} reversed />
                <YAxis tick={{ fontSize: 11, fill: "var(--muted-foreground)" }} tickLine={false} axisLine={false} orientation="right" />
                <Tooltip contentStyle={{ background: "var(--popover)", border: "1px solid var(--border)", borderRadius: 12, fontSize: 12 }} />
                <Area type="monotone" dataKey="reach" stroke="var(--brand-3)" strokeWidth={2} fill="url(#g2)" />
                <Area type="monotone" dataKey="engagement" stroke="var(--brand)" strokeWidth={2.4} fill="url(#g1)" />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="rounded-2xl border border-border bg-card shadow-card p-5">
          <h3 className="text-sm font-semibold">Content Mix</h3>
          <p className="text-xs text-muted-foreground">By post type</p>
          <div className="h-56 mt-2">
            <ResponsiveContainer>
              <PieChart>
                <Pie data={safeContentTypeData} dataKey="value" innerRadius={50} outerRadius={80} paddingAngle={3} stroke="none">
                  {safeContentTypeData.map((d, i) => <Cell key={i} fill={d.color} />)}
                </Pie>
                <Tooltip contentStyle={{ background: "var(--popover)", border: "1px solid var(--border)", borderRadius: 12, fontSize: 12 }} />
              </PieChart>
            </ResponsiveContainer>
          </div>
          <div className="space-y-2">
            {safeContentTypeData.map((d) => (
              <div key={d.name} className="flex items-center justify-between text-xs">
                <span className="flex items-center gap-2"><span className="h-2 w-2 rounded-full" style={{ background: d.color }} />{d.name}</span>
                <span className="num-display font-medium">{d.value}%</span>
              </div>
            ))}
          </div>
        </div>
      </div>

    </div>
  );
}

