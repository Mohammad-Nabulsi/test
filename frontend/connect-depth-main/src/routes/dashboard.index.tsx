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
  const [meBusiness, setMeBusiness] = useState<MeBusiness>(fallbackBusiness);
  const [meKpis, setMeKpis] = useState<MeKpi[]>(fallbackKpis as MeKpi[]);

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

      {/* Heatmap */}
      <div className="rounded-2xl border border-border bg-card shadow-card p-5">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h3 className="text-sm font-semibold">Best Posting Hours</h3>
            <p className="text-xs text-muted-foreground">Engagement intensity by day and hour</p>
          </div>
          <div className="flex items-center gap-1.5 text-[11px] text-muted-foreground">
            <span>Low</span>
            <div className="flex gap-0.5">
              {[0.15, 0.3, 0.45, 0.65, 0.85].map((o, i) => (<span key={i} className="h-3 w-3 rounded-sm" style={{ background: `oklch(0.55 0.22 277 / ${o})` }} />))}
            </div>
            <span>High</span>
          </div>
        </div>
        <div className="overflow-x-auto">
          <div className="min-w-[640px]">
            <div className="grid grid-cols-[60px_repeat(24,1fr)] gap-1 text-[10px] text-muted-foreground mb-1">
              <div></div>
              {Array.from({ length: 24 }, (_, h) => <div key={h} className="text-center">{h}</div>)}
            </div>
            {safeDays.map((day) => (
              <div key={day} className="grid grid-cols-[60px_repeat(24,1fr)] gap-1 mb-1">
                <div className="text-[11px] text-muted-foreground flex items-center">{day}</div>
                {safeHeatmapData.filter(d => d.day === day).map((d, i) => {
                  const o = Math.min(0.95, 0.08 + d.value / 110);
                  return <motion.div key={i} initial={{ opacity: 0, scale: 0.6 }} animate={{ opacity: 1, scale: 1 }} transition={{ delay: i * 0.005 }} className="h-7 rounded-md" style={{ background: `oklch(0.55 0.22 277 / ${o})` }} title={`${day} ${d.hour}:00 â€” ${d.value}`} />;
                })}
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Bottom Row */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <div className="lg:col-span-2 rounded-2xl border border-border bg-card shadow-card p-5">
          <h3 className="text-sm font-semibold mb-1">Top Hashtags by Engagement</h3>
          <p className="text-xs text-muted-foreground mb-3">Engagement rate per hashtag usage</p>
          <div className="h-60">
            <ResponsiveContainer>
              <BarChart data={safeTopHashtags} margin={{ left: -10 }}>
                <CartesianGrid stroke="var(--border)" strokeDasharray="3 3" vertical={false} />
                <XAxis dataKey="tag" tick={{ fontSize: 10, fill: "var(--muted-foreground)" }} tickLine={false} axisLine={false} angle={-15} height={48} reversed />
                <YAxis tick={{ fontSize: 11, fill: "var(--muted-foreground)" }} tickLine={false} axisLine={false} orientation="right" />
                <Tooltip cursor={{ fill: "var(--muted)" }} contentStyle={{ background: "var(--popover)", border: "1px solid var(--border)", borderRadius: 12, fontSize: 12 }} />
                <Bar dataKey="eng" radius={[8,8,0,0]} fill="var(--brand)" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="rounded-2xl border border-border bg-card shadow-card p-5">
          <div className="flex items-center gap-2 mb-3">
            <div className="h-7 w-7 rounded-lg bg-gradient-brand flex items-center justify-center"><Sparkles className="h-3.5 w-3.5 text-white" /></div>
            <div>
              <h3 className="text-sm font-semibold">Live Insights</h3>
              <p className="text-[11px] text-muted-foreground">AI-generated and continuously refreshed</p>
            </div>
          </div>
          <div className="space-y-2">
            {safeInsightStream.slice(0, 5).map((t, i) => (
              <motion.div key={i} initial={{ opacity: 0, x: 8 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: i * 0.15 }} className="flex gap-2 p-3 rounded-xl bg-muted/40 border border-border/60 text-xs leading-relaxed">
                <TrendingUp className="h-3.5 w-3.5 mt-0.5 text-[var(--brand)] shrink-0" />
                <span>{t}</span>
              </motion.div>
            ))}
          </div>
        </div>
      </div>

      {/* Top Posts This Week */}
      <div className="rounded-2xl border border-border bg-card shadow-card p-5">
        <h3 className="text-sm font-semibold mb-3">Top Posts This Week</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          {safePosts.slice(0, 4).map((p, i) => (
            <motion.div key={p.id} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: i * 0.06 }} className="group relative rounded-xl overflow-hidden border border-border">
              <div className="aspect-square relative" style={{ background: `linear-gradient(135deg, oklch(0.7 0.18 ${p.hue}), oklch(0.55 0.22 ${(p.hue + 60) % 360}))` }}>
                <div className="absolute inset-0 bg-gradient-to-t from-black/70 via-transparent to-transparent" />
                <div className="absolute top-2 right-2 px-1.5 py-0.5 text-[10px] uppercase tracking-wider rounded bg-black/40 text-white backdrop-blur-md">{p.typeLabel}</div>
                <div className="absolute bottom-2 left-2 right-2 text-white text-[11px] line-clamp-2">{p.caption}</div>
              </div>
              <div className="px-3 py-2 flex items-center justify-between text-[11px]">
                <span className="text-muted-foreground">{p.likes.toLocaleString("en-US")} likes</span>
                <span className="text-success font-medium">{p.engagement}%</span>
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    </div>
  );
}

