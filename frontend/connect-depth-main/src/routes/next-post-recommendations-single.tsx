import { createFileRoute } from "@tanstack/react-router";
import { motion } from "framer-motion";
import { AlertCircle, BadgePercent, MessageSquareMore, Sparkles, Target, TrendingUp } from "lucide-react";
import { useCallback, useMemo, useState } from "react";
import { KpiCard } from "@/components/dashboard/KpiCard";
import { SingleApiDatasetCard } from "@/components/dashboard/SingleApiDatasetCard";
import { SingleApiShell } from "@/components/dashboard/SingleApiShell";

export const Route = createFileRoute("/next-post-recommendations-single")({
  component: NextPostRecommendationsSinglePage,
});

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000";

type Summary = {
  total_recommendations?: number;
  estimated_total_impact?: number;
};

type BusinessBehaviorSummary = {
  posts_count?: number;
  business_name?: string;
  sector?: string;
  avg_hashtags_count?: number;
  avg_caption_length?: number;
  avg_emoji_count?: number;
  CTA_present_rate?: number;
  promo_post_rate?: number;
  mentions_location_rate?: number;
  pct_post_type_reel?: number;
};

type EngagementRecommendation = {
  id?: number;
  type?: string;
  category?: string;
  title?: string;
  explanation?: string;
  priority_score?: number;
  expected_impact?: number;
  confidence_score?: number;
  action_type?: string;
  icon?: string;
};

type NextPostRecommendationsResponse = {
  status: string;
  summary?: Summary;
  business_behavior_summary?: BusinessBehaviorSummary;
  engagement_recommendations?: EngagementRecommendation[];
};

function formatPercent(value?: number) {
  const safe = typeof value === "number" ? value : 0;
  return `${Math.round(safe * 100)}%`;
}

function priorityMeta(score?: number) {
  const safe = typeof score === "number" ? score : 0;
  if (safe >= 80) return { label: "أولوية عالية", color: "bg-destructive/10 text-destructive border-destructive/20" };
  if (safe >= 60) return { label: "أولوية متوسطة", color: "bg-amber-500/10 text-amber-700 border-amber-500/20" };
  return { label: "أولوية منخفضة", color: "bg-emerald-500/10 text-emerald-700 border-emerald-500/20" };
}

function healthMeta(label: string, value?: number) {
  const safe = typeof value === "number" ? value : 0;
  if (label === "CTA_present_rate") return safe >= 0.35 ? "قوة" : "ضعف";
  if (label === "promo_post_rate") return safe <= 0.55 ? "متوازن" : "مرتفع";
  if (label === "mentions_location_rate") return safe >= 0.35 ? "قوة" : "ضعف";
  if (label === "pct_post_type_reel") return safe >= 0.35 ? "قوة" : "ضعف";
  return "مؤشر";
}

function NextPostRecommendationsSinglePage() {
  const [datasetPath, setDatasetPath] = useState("vanilla_kpi_dataset.json");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<NextPostRecommendationsResponse | null>(null);

  const runRequest = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/api/next-post-recommendations-single`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ json_path: datasetPath }),
      });
      if (!res.ok) throw new Error(await res.text());
      setData((await res.json()) as NextPostRecommendationsResponse);
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "فشل تحميل توصيات المنشور القادم");
    } finally {
      setLoading(false);
    }
  }, [datasetPath]);

  const summary = data?.summary;
  const behavior = data?.business_behavior_summary;
  const recommendations = data?.engagement_recommendations ?? [];

  const behaviorCards = useMemo(() => ([
    { key: "posts_count", label: "عدد المنشورات", value: String(behavior?.posts_count ?? 0), tone: "مؤشر" },
    { key: "avg_hashtags_count", label: "متوسط الهاشتاغات", value: typeof behavior?.avg_hashtags_count === "number" ? behavior.avg_hashtags_count.toFixed(1) : "0.0", tone: "مؤشر" },
    { key: "avg_caption_length", label: "متوسط طول الكابشن", value: typeof behavior?.avg_caption_length === "number" ? behavior.avg_caption_length.toFixed(1) : "0.0", tone: "مؤشر" },
    { key: "avg_emoji_count", label: "متوسط الإيموجي", value: typeof behavior?.avg_emoji_count === "number" ? behavior.avg_emoji_count.toFixed(1) : "0.0", tone: "مؤشر" },
    { key: "CTA_present_rate", label: "وجود CTA", value: formatPercent(behavior?.CTA_present_rate), tone: healthMeta("CTA_present_rate", behavior?.CTA_present_rate) },
    { key: "promo_post_rate", label: "النبرة البيعية", value: formatPercent(behavior?.promo_post_rate), tone: healthMeta("promo_post_rate", behavior?.promo_post_rate) },
    { key: "mentions_location_rate", label: "ذكر الموقع", value: formatPercent(behavior?.mentions_location_rate), tone: healthMeta("mentions_location_rate", behavior?.mentions_location_rate) },
    { key: "pct_post_type_reel", label: "نسبة الريلز", value: formatPercent(behavior?.pct_post_type_reel), tone: healthMeta("pct_post_type_reel", behavior?.pct_post_type_reel) },
  ]), [behavior]);

  return (
    <SingleApiShell title="توصيات المنشور القادم" subtitle="لوحة ذكية لاختيار تحسينات المحتوى والتفاعل القادمة">
      <div className="space-y-5">
        <SingleApiDatasetCard
          apiBase={API_BASE}
          datasetPath={datasetPath}
          onDatasetPathChange={setDatasetPath}
          onRun={runRequest}
          runLabel="تشغيل Next Post Recommendations"
          running={loading}
          uploadNote="الطلب يرسل json_path مباشرة، والملف الافتراضي هو vanilla_kpi_dataset.json."
        />

        {error && <div className="rounded-2xl border border-destructive/30 bg-destructive/5 p-4 text-sm text-destructive">{error}</div>}

        {data && (
          <>
            <motion.section initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="relative overflow-hidden rounded-2xl border border-border bg-mesh p-6">
              <div className="absolute -top-20 -left-20 h-64 w-64 rounded-full bg-gradient-brand opacity-20 blur-3xl" />
              <div className="relative flex flex-col gap-5 lg:flex-row lg:items-end lg:justify-between">
                <div className="flex items-start gap-3">
                  <div className="h-11 w-11 rounded-xl bg-gradient-brand flex items-center justify-center shadow-glow shrink-0">
                    <Sparkles className="h-5 w-5 text-white" />
                  </div>
                  <div>
                    <h2 className="font-display text-2xl tracking-tight">ملخص استراتيجية المنشور القادم</h2>
                    <p className="text-sm text-muted-foreground mt-2 max-w-2xl">
                      {behavior?.business_name ? `اللوحة مبنية على سلوك ${behavior.business_name} الحالي لتحديد التوصيات الأكثر تأثيرًا على التفاعل.` : "اللوحة تعرض فرص التحسين الحالية للمحتوى والتفاعل."}
                    </p>
                  </div>
                </div>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 min-w-full lg:min-w-[360px]">
                  <KpiCard label="عدد التوصيات" value={summary?.total_recommendations ?? 0} delta={0} spark={[1, 2, 3, 3, 4, summary?.total_recommendations ?? 0]} />
                  <KpiCard label="الأثر الكلي المتوقع" value={summary?.estimated_total_impact ?? 0} suffix="%" delta={0} spark={[8, 12, 18, 24, 30, summary?.estimated_total_impact ?? 0]} />
                </div>
              </div>
            </motion.section>

            <motion.section initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} className="rounded-2xl border border-border bg-card shadow-card p-5">
              <div className="flex items-center justify-between gap-3 mb-4">
                <div>
                  <h3 className="text-sm font-semibold">ملخص سلوك البيزنس</h3>
                  <p className="text-xs text-muted-foreground mt-1">
                    {behavior?.business_name ?? "البيزنس"} {behavior?.sector ? `· ${behavior.sector}` : ""}
                  </p>
                </div>
                <div className="rounded-xl border border-border bg-background/60 px-3 py-2 text-xs text-muted-foreground">
                  مصدر القراءة: Business Behavior Summary
                </div>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-3">
                {behaviorCards.map((item, index) => (
                  <article key={item.key} className="rounded-2xl border border-border bg-background/70 p-4">
                    <div className="flex items-start justify-between gap-3">
                      <div>
                        <div className="text-xs text-muted-foreground">{item.label}</div>
                        <div className="mt-2 num-display text-2xl font-semibold">{item.value}</div>
                      </div>
                      <span className={`rounded-full border px-2.5 py-1 text-[11px] ${
                        item.tone === "قوة"
                          ? "bg-emerald-500/10 text-emerald-700 border-emerald-500/20"
                          : item.tone === "ضعف" || item.tone === "مرتفع"
                            ? "bg-destructive/10 text-destructive border-destructive/20"
                            : "bg-accent text-muted-foreground border-border"
                      }`}>
                        {item.tone}
                      </span>
                    </div>
                    <div className="mt-4 h-2 rounded-full bg-muted overflow-hidden">
                      <div
                        className={`h-full rounded-full ${
                          item.tone === "قوة"
                            ? "bg-emerald-500"
                            : item.tone === "ضعف" || item.tone === "مرتفع"
                              ? "bg-destructive"
                              : "bg-gradient-brand"
                        }`}
                        style={{ width: `${Math.min(100, Math.max(12, Number.parseFloat(item.value) || 12))}%` }}
                      />
                    </div>
                  </article>
                ))}
              </div>
            </motion.section>

            <motion.section initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} className="rounded-2xl border border-border bg-card shadow-card p-5">
              <div className="flex items-center justify-between gap-3 mb-4">
                <div>
                  <h3 className="text-sm font-semibold">توصيات التفاعل والمحتوى</h3>
                  <p className="text-xs text-muted-foreground mt-1">بطاقات مرتبة بصياغة تنفيذية تساعدك تختار الخطوة الجاية بسرعة.</p>
                </div>
                <div className="inline-flex items-center gap-2 rounded-xl border border-border bg-background/60 px-3 py-2 text-xs text-muted-foreground">
                  <TrendingUp className="h-3.5 w-3.5" />
                  {recommendations.length} توصية
                </div>
              </div>

              {recommendations.length ? (
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                  {recommendations.map((item, index) => {
                    const priority = priorityMeta(item.priority_score);
                    const progressWidth = Math.max(8, Math.min(100, Math.round(item.priority_score ?? 0)));
                    return (
                      <article key={`${item.id ?? index}`} className="rounded-2xl border border-border bg-background/70 p-5">
                        <div className="flex items-start justify-between gap-3">
                          <div className="flex-1 min-w-0">
                            <div className="flex flex-wrap items-center gap-2 mb-2">
                              <span className={`rounded-full border px-2.5 py-1 text-[11px] ${priority.color}`}>{priority.label}</span>
                              {item.category && <span className="rounded-full border border-border px-2.5 py-1 text-[11px] text-muted-foreground">{item.category}</span>}
                              {item.type && <span className="rounded-full border border-border px-2.5 py-1 text-[11px] text-muted-foreground">{item.type}</span>}
                            </div>
                            <h4 className="text-base font-semibold">{item.title ?? "توصية"}</h4>
                            <p className="mt-3 text-sm leading-6 text-muted-foreground">{item.explanation ?? "لا يوجد شرح متاح."}</p>
                          </div>
                          <div className="rounded-xl bg-accent/60 p-2 text-[var(--brand)]">
                            {item.icon === "sparkles" ? <Sparkles className="h-4 w-4" /> : item.action_type?.includes("engagement") ? <MessageSquareMore className="h-4 w-4" /> : <Target className="h-4 w-4" />}
                          </div>
                        </div>

                        <div className="mt-4 grid grid-cols-2 xl:grid-cols-4 gap-2 text-[11px]">
                          <div className="rounded-xl border border-border bg-card px-3 py-2">
                            <div className="text-muted-foreground">Priority Score</div>
                            <div className="num-display mt-1 font-semibold">{Math.round(item.priority_score ?? 0)}</div>
                          </div>
                          <div className="rounded-xl border border-border bg-card px-3 py-2">
                            <div className="text-muted-foreground">Expected Impact</div>
                            <div className="num-display mt-1 font-semibold">{Math.round(item.expected_impact ?? 0)}%</div>
                          </div>
                          <div className="rounded-xl border border-border bg-card px-3 py-2">
                            <div className="text-muted-foreground">Confidence</div>
                            <div className="num-display mt-1 font-semibold">{Math.round(item.confidence_score ?? 0)}%</div>
                          </div>
                          <div className="rounded-xl border border-border bg-card px-3 py-2">
                            <div className="text-muted-foreground">Action</div>
                            <div className="mt-1 font-semibold truncate">{item.action_type ?? "-"}</div>
                          </div>
                        </div>

                        <div className="mt-4">
                          <div className="flex items-center justify-between text-[11px] mb-1.5">
                            <span className="text-muted-foreground">ترتيب الأولوية</span>
                            <span className="num-display font-semibold">{progressWidth}%</span>
                          </div>
                          <div className="h-2 rounded-full bg-muted overflow-hidden">
                            <div className={`h-full rounded-full ${
                              progressWidth >= 80 ? "bg-destructive" : progressWidth >= 60 ? "bg-amber-500" : "bg-emerald-500"
                            }`} style={{ width: `${progressWidth}%` }} />
                          </div>
                        </div>
                      </article>
                    );
                  })}
                </div>
              ) : (
                <div className="rounded-2xl border border-dashed border-border bg-background/40 p-8 text-center text-sm text-muted-foreground">
                  <AlertCircle className="h-5 w-5 mx-auto mb-2" />
                  ما في توصيات راجعة حالياً من الـ API.
                </div>
              )}
            </motion.section>
          </>
        )}

        {!data && !loading && !error && (
          <div className="rounded-2xl border border-dashed border-border bg-card/70 p-8 text-center text-sm text-muted-foreground">
            شغّل الصفحة حتى تعرض لوحة توصيات المنشور القادم بشكل ديناميكي.
          </div>
        )}
      </div>
    </SingleApiShell>
  );
}
