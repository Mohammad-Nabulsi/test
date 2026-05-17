import { createFileRoute } from "@tanstack/react-router";
import { motion } from "framer-motion";
import { Sparkles, Store, Users } from "lucide-react";
import { useCallback, useState } from "react";
import { SingleApiDatasetCard } from "@/components/dashboard/SingleApiDatasetCard";
import { SingleApiShell } from "@/components/dashboard/SingleApiShell";
import { KpiCard } from "@/components/dashboard/KpiCard";

export const Route = createFileRoute("/similar-business-recommendations-single")({
  component: SimilarBusinessRecommendationsSinglePage,
});

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000";

type SimilarBusiness = {
  business_name?: string;
  sector?: string;
  similarity_score?: number;
  success_score?: number;
  rank?: number;
};

type SimilarBusinessRecommendation = {
  title?: string;
  explanation?: string;
  priority_score?: number;
};

type SimilarBusinessCard = {
  id?: number;
  title?: string;
  explanation?: string;
  expected_impact?: number;
  confidence_score?: number;
  priority_score?: number;
  inspired_by?: string[];
};

type SimilarBusinessResponse = {
  status: string;
  hero_summary?: {
    title?: string;
    total_recommendations?: number;
    top_opportunity?: string;
    estimated_best_impact?: number;
  };
  target_business?: {
    business_name?: string;
    sector?: string;
    success_score?: number;
  };
  similar_businesses?: SimilarBusiness[];
  benchmark_recommendations?: SimilarBusinessRecommendation[];
  recommendation_cards?: SimilarBusinessCard[];
  insight_highlights?: string[];
};

function SimilarBusinessRecommendationsSinglePage() {
  const [datasetPath, setDatasetPath] = useState("data/vanilla_kpi_dataset.json");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<SimilarBusinessResponse | null>(null);

  const runRequest = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/api/similar-business-recommendations-single`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ json_path: datasetPath }),
      });
      if (!res.ok) {
        throw new Error(await res.text());
      }
      setData((await res.json()) as SimilarBusinessResponse);
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "فشل تحميل التوصيات");
    } finally {
      setLoading(false);
    }
  }, [datasetPath]);

  const hero = data?.hero_summary;
  const target = data?.target_business;
  const peers = data?.similar_businesses ?? [];
  const benchmarkRecommendations = data?.benchmark_recommendations ?? [];
  const recommendationCards = data?.recommendation_cards ?? [];
  const highlights = data?.insight_highlights ?? [];

  return (
    <SingleApiShell title="توصيات البيزنسات المشابهة" subtitle="رؤى ديناميكية مبنية على أقرب الأنشطة لملفك الحالي">
      <div className="space-y-5">
        <SingleApiDatasetCard
          apiBase={API_BASE}
          datasetPath={datasetPath}
          onDatasetPathChange={setDatasetPath}
          onRun={runRequest}
          runLabel="تشغيل Similar Business Recommendations"
          running={loading}
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
                    <h2 className="font-display text-2xl tracking-tight">{hero?.title ?? "ملخص التوصيات"}</h2>
                    <p className="text-sm text-muted-foreground mt-2 max-w-2xl">
                      {hero?.top_opportunity ? `أقوى فرصة حالياً: ${hero.top_opportunity}.` : "تم تجهيز ملخص سريع يوضح أهم فرص التحسين بناءً على البيزنسات المشابهة."}
                    </p>
                  </div>
                </div>
                <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 min-w-full lg:min-w-[520px]">
                  <KpiCard label="عدد التوصيات" value={hero?.total_recommendations ?? 0} delta={0} spark={[2, 3, 3, 4, 5, hero?.total_recommendations ?? 0]} />
                  <KpiCard label="أفضل أثر متوقع" value={hero?.estimated_best_impact ?? 0} suffix="%" delta={0} spark={[8, 11, 13, 16, 20, hero?.estimated_best_impact ?? 0]} />
                  <KpiCard label="نجاح البيزنس" value={target?.success_score ?? 0} delta={0} spark={[20, 30, 38, 45, 56, target?.success_score ?? 0]} />
                </div>
              </div>
            </motion.section>

            <div className="grid grid-cols-1 xl:grid-cols-[360px,1fr] gap-4">
              <motion.section initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} className="rounded-2xl border border-border bg-card shadow-card p-5">
                <div className="flex items-center gap-2 mb-3">
                  <Store className="h-4 w-4 text-[var(--brand)]" />
                  <h3 className="text-sm font-semibold">نظرة على البيزنس المستهدف</h3>
                </div>
                <div className="space-y-3">
                  <div className="rounded-xl border border-border bg-background/60 p-4">
                    <div className="text-xs text-muted-foreground">اسم البيزنس</div>
                    <div className="mt-1 text-base font-semibold">{target?.business_name ?? "غير متوفر"}</div>
                  </div>
                  <div className="rounded-xl border border-border bg-background/60 p-4">
                    <div className="text-xs text-muted-foreground">القطاع</div>
                    <div className="mt-1 text-sm font-medium">{target?.sector ?? "غير متوفر"}</div>
                  </div>
                  <div className="rounded-xl border border-border bg-background/60 p-4">
                    <div className="text-xs text-muted-foreground">Success Score</div>
                    <div className="mt-1 num-display text-2xl font-semibold">{typeof target?.success_score === "number" ? target.success_score.toFixed(2) : "0.00"}</div>
                  </div>
                </div>
              </motion.section>

              <motion.section initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} className="rounded-2xl border border-border bg-card shadow-card p-5">
                <div className="flex items-center justify-between mb-3">
                  <div>
                    <h3 className="text-sm font-semibold">بيزنسات مشابهة</h3>
                    <p className="text-xs text-muted-foreground">عرض قابل للتمرير لأقرب المشاريع حسب التشابه والأداء</p>
                  </div>
                  <Users className="h-4 w-4 text-[var(--brand)]" />
                </div>
                {peers.length ? (
                  <div className="flex gap-3 overflow-x-auto pb-2">
                    {peers.map((peer, index) => {
                      const similarity = Math.round(peer.similarity_score ?? 0);
                      const success = Math.round(peer.success_score ?? 0);
                      return (
                        <article key={`${peer.business_name ?? "peer"}-${index}`} className="shrink-0 w-[270px] rounded-2xl border border-border bg-background/60 p-4">
                          <div className="flex items-center justify-between gap-3">
                            <div className="min-w-0">
                              <h4 className="text-sm font-semibold truncate">{peer.business_name ?? "بدون اسم"}</h4>
                              <p className="text-[11px] text-muted-foreground mt-1">{peer.sector ?? "بدون قطاع"}</p>
                            </div>
                            {peer.rank !== undefined && <span className="rounded-full bg-accent px-2.5 py-1 text-[11px] text-muted-foreground">#{peer.rank}</span>}
                          </div>
                          <div className="mt-4 space-y-3">
                            <div>
                              <div className="flex items-center justify-between text-[11px] mb-1">
                                <span className="text-muted-foreground">التشابه</span>
                                <span className="num-display font-semibold">{similarity}%</span>
                              </div>
                              <div className="h-2 rounded-full bg-muted overflow-hidden">
                                <div className="h-full bg-gradient-brand rounded-full" style={{ width: `${similarity}%` }} />
                              </div>
                            </div>
                            <div>
                              <div className="flex items-center justify-between text-[11px] mb-1">
                                <span className="text-muted-foreground">النجاح</span>
                                <span className="num-display font-semibold">{success}</span>
                              </div>
                              <div className="h-2 rounded-full bg-muted overflow-hidden">
                                <div className="h-full bg-[var(--brand-3)] rounded-full" style={{ width: `${Math.max(0, Math.min(100, success))}%` }} />
                              </div>
                            </div>
                          </div>
                        </article>
                      );
                    })}
                  </div>
                ) : (
                  <div className="rounded-xl border border-dashed border-border bg-background/40 p-5 text-sm text-muted-foreground">
                    ما رجع في بيزنسات مشابهة بهالاستجابة.
                  </div>
                )}
              </motion.section>
            </div>

            <motion.section initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} className="rounded-2xl border border-border bg-card shadow-card p-5">
              <h3 className="text-sm font-semibold mb-4">التوصيات المرجعية</h3>
              {benchmarkRecommendations.length ? (
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                  {benchmarkRecommendations.map((item, index) => (
                    <article key={`${item.title ?? "recommendation"}-${index}`} className="rounded-2xl border border-border bg-background/70 p-5">
                      <div className="flex items-start justify-between gap-3">
                        <h4 className="text-base font-semibold">{item.title ?? "توصية"}</h4>
                        <span className="rounded-full bg-accent px-2.5 py-1 text-[11px] text-muted-foreground">أولوية {Math.round(item.priority_score ?? 0)}</span>
                      </div>
                      <p className="mt-3 text-sm leading-6 text-muted-foreground">{item.explanation ?? "لا يوجد شرح متاح."}</p>
                    </article>
                  ))}
                </div>
              ) : (
                <div className="rounded-xl border border-dashed border-border bg-background/40 p-5 text-sm text-muted-foreground">
                  لا توجد benchmark recommendations حالياً.
                </div>
              )}
            </motion.section>

            <motion.section initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} className="rounded-2xl border border-border bg-card shadow-card p-5">
              <h3 className="text-sm font-semibold mb-4">بطاقات التوصيات</h3>
              {recommendationCards.length ? (
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                  {recommendationCards.map((card, index) => (
                    <article key={`${card.id ?? index}`} className="rounded-2xl border border-border bg-background/70 p-5">
                      <div className="flex flex-wrap items-start justify-between gap-3">
                        <div>
                          <h4 className="text-base font-semibold">{card.title ?? "بطاقة توصية"}</h4>
                          <p className="mt-3 text-sm leading-6 text-muted-foreground">{card.explanation ?? "لا يوجد شرح متاح."}</p>
                        </div>
                        <div className="flex flex-wrap gap-2 text-[11px]">
                          {typeof card.priority_score === "number" && <span className="rounded-full border border-border px-2.5 py-1">أولوية {Math.round(card.priority_score)}</span>}
                          {typeof card.expected_impact === "number" && <span className="rounded-full border border-border px-2.5 py-1">أثر {card.expected_impact}%</span>}
                          {typeof card.confidence_score === "number" && <span className="rounded-full border border-border px-2.5 py-1">ثقة {card.confidence_score}%</span>}
                        </div>
                      </div>
                      {!!card.inspired_by?.length && (
                        <div className="mt-4 flex flex-wrap gap-2">
                          {card.inspired_by.map((name) => (
                            <span key={name} className="rounded-full bg-accent/50 px-2.5 py-1 text-[11px] text-muted-foreground">{name}</span>
                          ))}
                        </div>
                      )}
                    </article>
                  ))}
                </div>
              ) : (
                <div className="rounded-xl border border-dashed border-border bg-background/40 p-5 text-sm text-muted-foreground">
                  لا توجد recommendation cards حالياً.
                </div>
              )}
            </motion.section>

            <motion.section initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} className="rounded-2xl border border-border bg-card shadow-card p-5">
              <h3 className="text-sm font-semibold mb-4">أبرز الإضاءات</h3>
              {highlights.length ? (
                <div className="flex flex-wrap gap-2">
                  {highlights.map((highlight, index) => (
                    <span key={`${highlight}-${index}`} className="rounded-full border border-border bg-gradient-soft px-3 py-1.5 text-xs">
                      {highlight}
                    </span>
                  ))}
                </div>
              ) : (
                <div className="rounded-xl border border-dashed border-border bg-background/40 p-5 text-sm text-muted-foreground">
                  لا توجد insight highlights حالياً.
                </div>
              )}
            </motion.section>
          </>
        )}

        {!data && !loading && !error && (
          <div className="rounded-2xl border border-dashed border-border bg-card/70 p-8 text-center text-sm text-muted-foreground">
            شغّل الصفحة حتى تعرض توصيات البيزنسات المشابهة بشكل ديناميكي.
          </div>
        )}
      </div>
    </SingleApiShell>
  );
}
