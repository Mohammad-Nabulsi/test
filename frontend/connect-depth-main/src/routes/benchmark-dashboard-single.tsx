import { createFileRoute } from "@tanstack/react-router";
import { motion } from "framer-motion";
import { PolarAngleAxis, PolarGrid, Radar, RadarChart, ResponsiveContainer, Tooltip } from "recharts";
import { Trophy } from "lucide-react";
import { useCallback, useMemo, useState } from "react";
import { SingleApiDatasetCard } from "@/components/dashboard/SingleApiDatasetCard";
import { SingleApiShell } from "@/components/dashboard/SingleApiShell";
import { KpiCard } from "@/components/dashboard/KpiCard";

export const Route = createFileRoute("/benchmark-dashboard-single")({
  component: BenchmarkDashboardSinglePage,
});

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000";

type BenchmarkSummary = {
  business_name?: string;
  sector?: string;
  sector_rank?: number;
  sector_percentile?: number;
  business_score?: number;
  total_sector_businesses?: number;
  summary_text?: string;
};

type SectorRankingRow = {
  rank?: number;
  business_name?: string;
  success_score?: number;
};

type RadarChartPayload = {
  labels?: string[];
  business_values?: number[];
  sector_average_values?: number[];
};

type KpiComparison = {
  metric_name?: string;
  business_value?: number;
  sector_average?: number;
  top_sector_value?: number;
  formatted_text?: string;
  gpt_insight?: string;
};

type BenchmarkDashboardResponse = {
  status: string;
  business_summary?: BenchmarkSummary;
  sector_ranking?: SectorRankingRow[];
  radar_chart?: RadarChartPayload;
  kpi_comparisons?: KpiComparison[];
  sector_insights?: string[];
};

function BenchmarkDashboardSinglePage() {
  const [datasetPath, setDatasetPath] = useState("data/vanilla_kpi_dataset.json");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<BenchmarkDashboardResponse | null>(null);

  const runRequest = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/api/benchmark-dashboard-single`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ json_path: datasetPath }),
      });
      if (!res.ok) {
        throw new Error(await res.text());
      }
      setData((await res.json()) as BenchmarkDashboardResponse);
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "فشل تحميل benchmark dashboard");
    } finally {
      setLoading(false);
    }
  }, [datasetPath]);

  const summary = data?.business_summary;
  const ranking = data?.sector_ranking ?? [];
  const comparisons = data?.kpi_comparisons ?? [];
  const insights = data?.sector_insights ?? [];
  const radarData = useMemo(() => {
    const chart = data?.radar_chart;
    const labels = chart?.labels ?? [];
    const businessValues = chart?.business_values ?? [];
    const sectorValues = chart?.sector_average_values ?? [];
    return labels.map((label, index) => ({
      metric: label,
      business: businessValues[index] ?? 0,
      sector: sectorValues[index] ?? 0,
    }));
  }, [data?.radar_chart]);

  return (
    <SingleApiShell title="Benchmark Dashboard" subtitle="لوحة مقارنة ديناميكية بين البيزنس الحالي ومتوسط القطاع">
      <div className="space-y-5">
        <SingleApiDatasetCard
          apiBase={API_BASE}
          datasetPath={datasetPath}
          onDatasetPathChange={setDatasetPath}
          onRun={runRequest}
          runLabel="تشغيل Benchmark Dashboard"
          running={loading}
        />

        {error && <div className="rounded-2xl border border-destructive/30 bg-destructive/5 p-4 text-sm text-destructive">{error}</div>}

        {data && (
          <>
            <motion.section initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="rounded-2xl border border-border p-6 bg-mesh">
              <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-5">
                <div>
                  <div className="text-xs uppercase tracking-widest text-muted-foreground">ملخص benchmark</div>
                  <div className="flex items-baseline gap-2 mt-2">
                    <span className="num-display text-5xl font-semibold text-gradient-brand">{summary?.sector_percentile ?? 0}</span>
                    <span className="text-sm text-muted-foreground">من 100</span>
                  </div>
                  <p className="text-sm mt-2 max-w-2xl">{summary?.summary_text ?? "تم تجهيز ملخص المقارنة."}</p>
                </div>
                <div className="relative h-32 w-32 self-center">
                  <svg viewBox="0 0 100 100" className="h-full w-full -rotate-90">
                    <circle cx="50" cy="50" r="42" fill="none" stroke="var(--muted)" strokeWidth="8" />
                    <circle
                      cx="50"
                      cy="50"
                      r="42"
                      fill="none"
                      stroke="url(#benchmarkGrad)"
                      strokeWidth="8"
                      strokeLinecap="round"
                      strokeDasharray={`${((summary?.sector_percentile ?? 0) / 100) * 264} 264`}
                    />
                    <defs>
                      <linearGradient id="benchmarkGrad" x1="0" x2="1">
                        <stop offset="0%" stopColor="var(--brand)" />
                        <stop offset="100%" stopColor="var(--brand-3)" />
                      </linearGradient>
                    </defs>
                  </svg>
                  <div className="absolute inset-0 flex items-center justify-center">
                    <Trophy className="h-8 w-8 text-[var(--brand)]" />
                  </div>
                </div>
              </div>
            </motion.section>

            <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-4">
              <KpiCard label="ترتيب القطاع" value={summary?.sector_rank ?? 0} delta={0} spark={[1, 1, 2, 2, 3, summary?.sector_rank ?? 0]} />
              <KpiCard label="المئين" value={summary?.sector_percentile ?? 0} suffix="%" delta={0} spark={[10, 24, 39, 52, 64, summary?.sector_percentile ?? 0]} />
              <KpiCard label="Business Score" value={summary?.business_score ?? 0} delta={0} spark={[12, 18, 25, 34, 48, summary?.business_score ?? 0]} />
              <KpiCard label="عدد شركات القطاع" value={summary?.total_sector_businesses ?? 0} delta={0} spark={[2, 4, 5, 7, 8, summary?.total_sector_businesses ?? 0]} />
            </div>

            <div className="grid grid-cols-1 xl:grid-cols-[1.3fr,0.9fr] gap-4">
              <motion.section initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} className="rounded-2xl border border-border bg-card shadow-card p-5">
                <h3 className="text-sm font-semibold">مقارنة متعددة المحاور</h3>
                <p className="text-xs text-muted-foreground mb-3">البيزنس الحالي مقابل متوسط القطاع مباشرة من بيانات الـ API</p>
                {radarData.length ? (
                  <div className="h-80">
                    <ResponsiveContainer>
                      <RadarChart data={radarData}>
                        <PolarGrid stroke="var(--border)" />
                        <PolarAngleAxis dataKey="metric" tick={{ fontSize: 11, fill: "var(--muted-foreground)" }} />
                        <Tooltip contentStyle={{ background: "var(--popover)", border: "1px solid var(--border)", borderRadius: 12, fontSize: 12 }} />
                        <Radar name="البيزنس" dataKey="business" stroke="var(--brand)" fill="var(--brand)" fillOpacity={0.35} strokeWidth={2} />
                        <Radar name="القطاع" dataKey="sector" stroke="var(--brand-3)" fill="var(--brand-3)" fillOpacity={0.18} strokeWidth={2} />
                      </RadarChart>
                    </ResponsiveContainer>
                  </div>
                ) : (
                  <div className="rounded-xl border border-dashed border-border bg-background/40 p-5 text-sm text-muted-foreground">
                    لا توجد بيانات كافية للرادار.
                  </div>
                )}
              </motion.section>

              <motion.section initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} className="rounded-2xl border border-border bg-card shadow-card p-5">
                <h3 className="text-sm font-semibold mb-3">ترتيب القطاع</h3>
                {ranking.length ? (
                  <div className="space-y-2.5">
                    {ranking.map((item, index) => {
                      const isCurrentBusiness = item.business_name === summary?.business_name;
                      return (
                        <div key={`${item.business_name ?? "row"}-${index}`} className={`flex items-center gap-3 rounded-xl p-3 ${isCurrentBusiness ? "bg-gradient-soft border border-[var(--brand)]/30" : "bg-muted/40"}`}>
                          <div className="num-display text-lg font-semibold text-muted-foreground w-6">{item.rank ?? index + 1}</div>
                          <div className="flex-1 min-w-0">
                            <div className="text-sm font-medium truncate">
                              {item.business_name ?? "بدون اسم"}{" "}
                              {isCurrentBusiness && <span className="text-[10px] px-1.5 py-0.5 rounded bg-gradient-brand text-white">أنت</span>}
                            </div>
                          </div>
                          <div className="num-display text-sm font-semibold">{typeof item.success_score === "number" ? item.success_score.toFixed(2) : "0.00"}</div>
                        </div>
                      );
                    })}
                  </div>
                ) : (
                  <div className="rounded-xl border border-dashed border-border bg-background/40 p-5 text-sm text-muted-foreground">
                    لا يوجد sector ranking حالياً.
                  </div>
                )}
              </motion.section>
            </div>

            <motion.section initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} className="rounded-2xl border border-border bg-card shadow-card p-5">
              <h3 className="text-sm font-semibold mb-4">تفصيل المؤشرات</h3>
              {comparisons.length ? (
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                  {comparisons.map((item, index) => (
                    <article key={`${item.metric_name ?? "metric"}-${index}`} className="rounded-2xl border border-border bg-background/70 p-5">
                      <div className="flex items-center justify-between gap-3">
                        <h4 className="text-base font-semibold">{item.metric_name ?? "Metric"}</h4>
                        <span className="rounded-full bg-accent px-2.5 py-1 text-[11px] text-muted-foreground">
                          الأعلى {typeof item.top_sector_value === "number" ? item.top_sector_value.toFixed(2) : "0.00"}
                        </span>
                      </div>
                      <div className="mt-4 grid grid-cols-3 gap-2">
                        <div className="rounded-xl border border-border bg-card p-3">
                          <div className="text-[11px] text-muted-foreground">البيزنس</div>
                          <div className="num-display text-lg font-semibold mt-1">{typeof item.business_value === "number" ? item.business_value.toFixed(2) : "0.00"}</div>
                        </div>
                        <div className="rounded-xl border border-border bg-card p-3">
                          <div className="text-[11px] text-muted-foreground">متوسط القطاع</div>
                          <div className="num-display text-lg font-semibold mt-1">{typeof item.sector_average === "number" ? item.sector_average.toFixed(2) : "0.00"}</div>
                        </div>
                        <div className="rounded-xl border border-border bg-card p-3">
                          <div className="text-[11px] text-muted-foreground">أفضل قيمة</div>
                          <div className="num-display text-lg font-semibold mt-1">{typeof item.top_sector_value === "number" ? item.top_sector_value.toFixed(2) : "0.00"}</div>
                        </div>
                      </div>
                      <p className="mt-4 text-sm font-medium leading-6">{item.formatted_text ?? "لا يوجد وصف مختصر."}</p>
                      <p className="mt-3 text-sm leading-6 text-muted-foreground">{item.gpt_insight ?? "لا يوجد insight تفصيلي."}</p>
                    </article>
                  ))}
                </div>
              ) : (
                <div className="rounded-xl border border-dashed border-border bg-background/40 p-5 text-sm text-muted-foreground">
                  لا توجد KPI comparisons حالياً.
                </div>
              )}
            </motion.section>

            <motion.section initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} className="rounded-2xl border border-border bg-card shadow-card p-5">
              <h3 className="text-sm font-semibold mb-4">إضاءات استراتيجية</h3>
              {insights.length ? (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                  {insights.map((insight, index) => (
                    <article key={`${insight}-${index}`} className="rounded-xl border border-border bg-gradient-soft p-4 text-sm leading-6">
                      {insight}
                    </article>
                  ))}
                </div>
              ) : (
                <div className="rounded-xl border border-dashed border-border bg-background/40 p-5 text-sm text-muted-foreground">
                  لا توجد sector insights حالياً.
                </div>
              )}
            </motion.section>
          </>
        )}

        {!data && !loading && !error && (
          <div className="rounded-2xl border border-dashed border-border bg-card/70 p-8 text-center text-sm text-muted-foreground">
            شغّل الصفحة حتى تعرض benchmark dashboard بشكل ديناميكي.
          </div>
        )}
      </div>
    </SingleApiShell>
  );
}
