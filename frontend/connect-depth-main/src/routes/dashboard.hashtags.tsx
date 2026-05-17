import { createFileRoute } from "@tanstack/react-router";
import { motion } from "framer-motion";
import { AlertTriangle, CheckCircle2, Hash, Sparkles, UploadCloud } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { type CaptionAnalysisResult, getLatestCaptionAnalysis } from "@/lib/upload-state";

export const Route = createFileRoute("/dashboard/hashtags")({
  component: HashtagTipsPage,
});

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000";

type HashtagRule = {
  hashtags_clean?: string;
  hashtag?: string;
  recommendation_text?: string;
  confidence?: number;
  lift?: number;
  count?: number;
  antecedent_count?: number;
  reliability?: string;
};

type HashtagResponse = {
  top_recommendations?: HashtagRule[];
  warning_recommendations?: HashtagRule[];
  used_fallback?: boolean;
  summary?: Array<{ metric?: string; value?: unknown }>;
};

type TopicStageResponse = {
  summary?: Record<string, unknown>;
  insight_cards?: Array<Record<string, unknown>>;
  topic_recommendations?: Array<Record<string, unknown>>;
  fit_attempts?: Array<Record<string, unknown>>;
  visualizations?: {
    available?: boolean;
    files?: Record<string, string>;
    errors?: Array<{ view?: string; error?: string }>;
  };
};

type ApiCallState = {
  ok: boolean;
  data?: unknown;
  error?: string;
};

type RunState = {
  datasetId: string;
  preprocessKpi: ApiCallState;
  runHashtag: ApiCallState;
  runTopic: ApiCallState;
  getHashtag: ApiCallState;
  getTopic: ApiCallState;
};

function formatPercent(value?: number) {
  if (typeof value !== "number" || Number.isNaN(value)) return "N/A";
  return `${Math.round(value * 100)}%`;
}

function normalizeTagList(rule: HashtagRule) {
  const raw = rule.hashtags_clean || rule.hashtag || "";
  return raw
    .split(",")
    .map((tag) => tag.trim())
    .filter(Boolean)
    .map((tag) => (tag.startsWith("#") ? tag : `#${tag}`));
}

function metricValue(summary: HashtagResponse["summary"], metric: string) {
  return summary?.find((item) => item.metric === metric)?.value;
}

function pickTopicPill(row: Record<string, unknown>) {
  const label = String(row.business_label ?? row.topic_label ?? row.label ?? row.topic_name ?? "").trim();
  if (label) return label;
  const topicId = row.topic_id;
  if (topicId !== undefined && topicId !== null && String(topicId).trim()) {
    return `Topic ${String(topicId)}`;
  }
  return "غير متوفر";
}

async function parseResponse(response: Response) {
  const contentType = response.headers.get("content-type") || "";
  if (contentType.includes("application/json")) return response.json();
  return response.text();
}

async function callApi(url: string, init?: RequestInit): Promise<ApiCallState> {
  try {
    const response = await fetch(url, init);
    const payload = await parseResponse(response);
    if (!response.ok) {
      const text = typeof payload === "string" ? payload : JSON.stringify(payload);
      return { ok: false, error: text };
    }
    return { ok: true, data: payload };
  } catch (error) {
    return { ok: false, error: error instanceof Error ? error.message : "Network error" };
  }
}

function HashtagTipsPage() {
  const [analysis, setAnalysis] = useState<CaptionAnalysisResult | null>(null);
  const [datasetIdInput, setDatasetIdInput] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [runState, setRunState] = useState<RunState | null>(null);
  const [running, setRunning] = useState(false);
  const [runError, setRunError] = useState<string | null>(null);
  const [vizView, setVizView] = useState<"intertopic" | "barchart" | "heatmap">("intertopic");

  useEffect(() => {
    const latest = getLatestCaptionAnalysis();
    setAnalysis(latest);
    if (latest?.dataset_id) setDatasetIdInput(latest.dataset_id);
  }, []);

  const hashtagData =
    (runState?.getHashtag.data as HashtagResponse | undefined) ??
    (analysis?.hashtag_stage as HashtagResponse | undefined);

  const topicData = (runState?.getTopic.data as TopicStageResponse | undefined) ?? null;
  const useTags = useMemo(() => (hashtagData?.top_recommendations ?? []).slice(0, 10), [hashtagData]);
  const avoidTags = useMemo(() => (hashtagData?.warning_recommendations ?? []).slice(0, 8), [hashtagData]);
  const hasData = Boolean(hashtagData);
  const adaptiveMinCount = metricValue(hashtagData?.summary, "adaptive_min_count");
  const activeDatasetId = runState?.datasetId || datasetIdInput || analysis?.dataset_id || "";

  const insightCards = (topicData?.insight_cards ?? []).slice(0, 12);
  const topicRecs = (topicData?.topic_recommendations ?? []).slice(0, 8);
  const fitMethod = String(topicData?.summary?.fit_method ?? "");
  const vizFiles = (topicData?.visualizations?.files ?? {}) as Record<string, string>;
  const hasVizMeta = Boolean(topicData?.visualizations);
  const availableVizViews = useMemo(() => {
    const order: Array<"intertopic" | "barchart" | "heatmap"> = ["intertopic", "barchart", "heatmap"];
    return order.filter((view) => Boolean(vizFiles[view]));
  }, [vizFiles]);
  const isVizViewAvailable = (view: "intertopic" | "barchart" | "heatmap") => availableVizViews.includes(view);
  const selectedVizError = (topicData?.visualizations?.errors ?? []).find((item) => item?.view === vizView)?.error;
  const visualizationUrl = activeDatasetId
    ? `${API_BASE}/api/datasets/${encodeURIComponent(activeDatasetId)}/business-topic-visualization?view=${vizView}`
    : "";

  useEffect(() => {
    if (!hasVizMeta) return;
    if (!availableVizViews.length) return;
    if (availableVizViews.includes(vizView)) return;
    setVizView(availableVizViews[0]);
  }, [hasVizMeta, availableVizViews, vizView]);

  async function runFromUpload(selectedFile: File) {
    setRunning(true);
    setRunError(null);
    setRunState(null);

    const preprocessForm = new FormData();
    preprocessForm.append("file", selectedFile);

    const preprocessKpi = await callApi(`${API_BASE}/api/datasets/upload-preprocess-kpi`, {
      method: "POST",
      body: preprocessForm,
    });

    if (!preprocessKpi.ok) {
      setRunning(false);
      setRunError(preprocessKpi.error || "فشل preprocess/kpi.");
      return;
    }

    const datasetId = String((preprocessKpi.data as { dataset_id?: string })?.dataset_id || "");
    if (!datasetId) {
      setRunning(false);
      setRunError("لم يتم العثور على dataset_id في الاستجابة.");
      return;
    }

    setDatasetIdInput(datasetId);

    const [runHashtag, runTopic] = await Promise.all([
      callApi(`${API_BASE}/api/datasets/${datasetId}/stages/hashtag`, { method: "POST" }),
      callApi(`${API_BASE}/api/datasets/${datasetId}/stages/business-topic-insights-local-cpu`, { method: "POST" }),
    ]);

    const [getHashtag, getTopic] = await Promise.all([
      callApi(`${API_BASE}/api/datasets/${datasetId}/hashtag-recommendations`),
      callApi(`${API_BASE}/api/datasets/${datasetId}/dynamic-ngram-insights`),
    ]);

    setRunState({ datasetId, preprocessKpi, runHashtag, runTopic, getHashtag, getTopic });

    if (!runHashtag.ok || !runTopic.ok || !getHashtag.ok || !getTopic.ok) {
      setRunError(runHashtag.error || runTopic.error || getHashtag.error || getTopic.error || "فشل تشغيل/جلب النتائج.");
    }

    setRunning(false);
  }

  async function refreshForDatasetId() {
    if (!datasetIdInput.trim()) {
      setRunError("أدخل dataset_id أولاً.");
      return;
    }

    setRunning(true);
    setRunError(null);
    const datasetId = datasetIdInput.trim();

    const [runHashtag, runTopic] = await Promise.all([
      callApi(`${API_BASE}/api/datasets/${datasetId}/stages/hashtag`, { method: "POST" }),
      callApi(`${API_BASE}/api/datasets/${datasetId}/stages/business-topic-insights-local-cpu`, { method: "POST" }),
    ]);

    const [getHashtag, getTopic] = await Promise.all([
      callApi(`${API_BASE}/api/datasets/${datasetId}/hashtag-recommendations`),
      callApi(`${API_BASE}/api/datasets/${datasetId}/dynamic-ngram-insights`),
    ]);

    setRunState({
      datasetId,
      preprocessKpi: { ok: true, data: { note: "refresh mode" } },
      runHashtag,
      runTopic,
      getHashtag,
      getTopic,
    });

    if (!runHashtag.ok || !runTopic.ok || !getHashtag.ok || !getTopic.ok) {
      setRunError(runHashtag.error || runTopic.error || getHashtag.error || getTopic.error || "فشل تحديث النتائج.");
    }

    setRunning(false);
  }

  return (
    <div className="space-y-6" dir="rtl">
      <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="relative overflow-hidden rounded-2xl border border-border bg-mesh p-5 md:p-6">
        <div className="relative flex flex-col gap-3">
          <div className="inline-flex w-fit items-center gap-2 rounded-full border border-border bg-card/80 px-3 py-1 text-xs font-medium text-muted-foreground backdrop-blur-md">
            <Sparkles className="h-3.5 w-3.5 text-[var(--brand)]" />
            تجميع المحتوى
          </div>
          <h2 className="font-display text-2xl tracking-tight md:text-3xl">تحليل ديناميكي للهاشتاقات وBERTopic</h2>
          <p className="max-w-2xl text-sm leading-relaxed text-muted-foreground">
            عند رفع الملف يتم تشغيل preprocess/KPI ثم hashtag recommendations وBERTopic المحلي على CPU، ثم عرض البطاقات والمرئيات مباشرة.
          </p>
        </div>
      </motion.div>

      <section className="rounded-2xl border border-border bg-card p-5 shadow-card space-y-4">
        <div className="flex flex-col gap-3 lg:flex-row lg:items-end">
          <div className="flex-1">
            <label className="mb-2 block text-xs text-muted-foreground">ارفع ملف .json أو .csv</label>
            <input
              type="file"
              accept=".csv,.json"
              onChange={(event) => setFile(event.target.files?.[0] ?? null)}
              className="block w-full rounded-xl border border-border bg-background px-3 py-2 text-sm"
            />
          </div>
          <button
            disabled={!file || running}
            onClick={() => file && runFromUpload(file)}
            className="inline-flex items-center gap-2 rounded-xl bg-[var(--brand)] px-4 py-2 text-sm font-medium text-white disabled:opacity-50"
          >
            <UploadCloud className="h-4 w-4" />
            {running ? "جاري التشغيل..." : "تشغيل التحليل الكامل"}
          </button>
        </div>

        <div className="flex flex-col gap-3 lg:flex-row lg:items-end">
          <div className="flex-1">
            <label className="mb-2 block text-xs text-muted-foreground">أو استخدم dataset_id موجودًا</label>
            <input
              value={datasetIdInput}
              onChange={(event) => setDatasetIdInput(event.target.value)}
              placeholder="مثال: UUID"
              className="block w-full rounded-xl border border-border bg-background px-3 py-2 text-sm"
            />
          </div>
          <button
            disabled={!datasetIdInput.trim() || running}
            onClick={refreshForDatasetId}
            className="rounded-xl border border-border bg-muted px-4 py-2 text-sm font-medium disabled:opacity-50"
          >
            {running ? "جاري التحديث..." : "تحديث النتائج"}
          </button>
        </div>

        {runError && <div className="rounded-xl border border-destructive/30 bg-destructive/5 p-3 text-xs text-destructive">{runError}</div>}
        <div className="text-xs text-muted-foreground">
          dataset_id النشط: <span className="font-medium text-foreground">{activeDatasetId || "غير متوفر"}</span>
        </div>
      </section>

      {!hasData && (
        <div className="rounded-2xl border border-border bg-card p-6 text-sm text-muted-foreground shadow-card">
          ارفع ملفًا لعرض بطاقات الهاشتاقات ونتائج BERTopic.
        </div>
      )}

      {hasData && (
        <>
          <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
            <SummaryCard icon={Hash} label="هاشتاقات مقترحة" value={useTags.length} hint={`أدنى حد تكيفي: ${adaptiveMinCount ?? "غير متوفر"}`} />
            <SummaryCard
              icon={AlertTriangle}
              label="هاشتاقات تحتاج حذرًا"
              value={avoidTags.length}
              hint={hashtagData?.used_fallback ? "تم استخدام بديل احتياطي" : "مستخرجة من قواعد الارتباط"}
            />
          </div>

          <section className="grid grid-cols-1 gap-4 xl:grid-cols-2">
            <div className="rounded-2xl border border-border bg-card p-5 shadow-card">
              <div className="mb-4 flex items-center gap-2">
                <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-success/10">
                  <CheckCircle2 className="h-4 w-4 text-success" />
                </div>
                <div>
                  <h3 className="text-sm font-semibold">استخدم هذه الهاشتاقات</h3>
                  <p className="text-xs text-muted-foreground">مرتبة حسب الثقة والرفع</p>
                </div>
              </div>
              <div className="space-y-3">
                {useTags.length ? useTags.map((rule, index) => <HashtagCard key={`u-${index}`} rule={rule} tone="use" />) : <EmptyState text="لا توجد توصيات استخدام واضحة." />}
              </div>
            </div>

            <div className="rounded-2xl border border-border bg-card p-5 shadow-card">
              <div className="mb-4 flex items-center gap-2">
                <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-destructive/10">
                  <AlertTriangle className="h-4 w-4 text-destructive" />
                </div>
                <div>
                  <h3 className="text-sm font-semibold">استخدم بحذر</h3>
                  <p className="text-xs text-muted-foreground">أنماط مرتبطة بنتائج أضعف</p>
                </div>
              </div>
              <div className="space-y-3">
                {avoidTags.length ? avoidTags.map((rule, index) => <HashtagCard key={`a-${index}`} rule={rule} tone="avoid" />) : <EmptyState text="لا توجد هاشتاقات تحذيرية." />}
              </div>
            </div>
          </section>
        </>
      )}

      <section className="rounded-2xl border border-border bg-card p-5 shadow-card space-y-4">
        <div className="flex items-center justify-between gap-3">
          <h3 className="text-base font-semibold">رؤى BERTopic الديناميكية</h3>
          <span className="rounded-full border border-border bg-muted px-2.5 py-1 text-xs text-muted-foreground">
            {fitMethod ? `fit_method: ${fitMethod}` : "غير متوفر"}
          </span>
        </div>

        {!!topicRecs.length && (
          <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
            {topicRecs.map((row, idx) => (
              <div key={`topic-rec-${idx}`} className="rounded-xl border border-border bg-gradient-soft p-4">
                <div className="text-sm font-semibold">{String(row.action ?? "Recommendation")}</div>
                <div className="mt-1 text-xs text-muted-foreground">{String(row.reason ?? "")}</div>
                <div className="mt-2 text-xs">
                  <span className="rounded-full bg-card px-2 py-1">{pickTopicPill(row)}</span>
                </div>
              </div>
            ))}
          </div>
        )}

        <div className="grid grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-3">
          {insightCards.length ? (
            insightCards.map((card, idx) => <InsightCard key={`insight-${idx}`} card={card} />)
          ) : (
            <EmptyState text="لا توجد بطاقات insights بعد. شغّل التحليل أولًا." />
          )}
        </div>

        <div className="rounded-xl border border-border p-4">
          <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
            <h4 className="text-sm font-semibold">تصور BERTopic</h4>
            <div className="flex items-center gap-2">
              <button
                onClick={() => setVizView("intertopic")}
                disabled={hasVizMeta && !isVizViewAvailable("intertopic")}
                className={`rounded-lg px-2.5 py-1 text-xs ${vizView === "intertopic" ? "bg-[var(--brand)] text-white" : "bg-muted"} ${hasVizMeta && !isVizViewAvailable("intertopic") ? "cursor-not-allowed opacity-50" : ""}`}
              >
                Intertopic
              </button>
              <button
                onClick={() => setVizView("barchart")}
                disabled={hasVizMeta && !isVizViewAvailable("barchart")}
                className={`rounded-lg px-2.5 py-1 text-xs ${vizView === "barchart" ? "bg-[var(--brand)] text-white" : "bg-muted"} ${hasVizMeta && !isVizViewAvailable("barchart") ? "cursor-not-allowed opacity-50" : ""}`}
              >
                Barchart
              </button>
              <button
                onClick={() => setVizView("heatmap")}
                disabled={hasVizMeta && !isVizViewAvailable("heatmap")}
                className={`rounded-lg px-2.5 py-1 text-xs ${vizView === "heatmap" ? "bg-[var(--brand)] text-white" : "bg-muted"} ${hasVizMeta && !isVizViewAvailable("heatmap") ? "cursor-not-allowed opacity-50" : ""}`}
              >
                Heatmap
              </button>
            </div>
          </div>

          {activeDatasetId ? (
            hasVizMeta && !availableVizViews.length ? (
              <EmptyState text={selectedVizError ? `تعذر إنشاء ${vizView}: ${selectedVizError}` : "لا توجد مرئيات BERTopic متاحة لهذه التشغيله."} />
            ) : selectedVizError && !isVizViewAvailable(vizView) ? (
              <EmptyState text={`تعذر إنشاء ${vizView}: ${selectedVizError}`} />
            ) : (
            <iframe
              src={visualizationUrl}
              title={`BERTopic ${vizView}`}
              className="h-[540px] w-full rounded-lg border border-border bg-background"
              loading="lazy"
            />
            )
          ) : (
            <EmptyState text="لا يوجد dataset_id لعرض visualization." />
          )}
        </div>
      </section>
    </div>
  );
}

function InsightCard({ card }: { card: Record<string, unknown> }) {
  const title = String(card.business_label ?? card.label ?? card.topic_label ?? card.title ?? "Insight");
  const insight = String(card.insight ?? card.insight_text ?? card.description ?? "");
  const suggestion = String(card.suggested_action ?? card.suggested_next_step ?? card.recommendation ?? "");

  return (
    <div className="rounded-xl border border-border bg-card p-4 shadow-card">
      <div className="text-sm font-semibold">{title}</div>
      {insight && <p className="mt-2 text-xs leading-relaxed text-muted-foreground">{insight}</p>}
      {suggestion && <div className="mt-3 rounded-lg border border-[var(--brand)]/20 bg-accent/40 p-2.5 text-xs">{suggestion}</div>}
    </div>
  );
}

function SummaryCard({ icon: Icon, label, value, hint }: { icon: React.ComponentType<{ className?: string }>; label: string; value: string | number; hint: string }) {
  return (
    <div className="rounded-2xl border border-border bg-card p-4 shadow-card">
      <div className="flex items-center gap-3">
        <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-accent">
          <Icon className="h-4 w-4 text-[var(--brand)]" />
        </div>
        <div>
          <div className="text-xs text-muted-foreground">{label}</div>
          <div className="num-display text-2xl font-semibold">{value}</div>
        </div>
      </div>
      <div className="mt-3 text-xs text-muted-foreground">{hint}</div>
    </div>
  );
}

function HashtagCard({ rule, tone }: { rule: HashtagRule; tone: "use" | "avoid" }) {
  const tags = normalizeTagList(rule);
  return (
    <div className={`rounded-xl border p-3 ${tone === "use" ? "border-success/20 bg-success/5" : "border-destructive/20 bg-destructive/5"}`}>
      <div className="flex flex-wrap gap-1.5">
        {tags.length ? tags.map((tag) => (
          <span key={tag} className="rounded-full bg-card px-2.5 py-1 text-xs font-medium shadow-sm">{tag}</span>
        )) : <span className="text-sm font-medium">لا توجد قيمة هاشتاق</span>}
      </div>
      <p className="mt-2 text-xs leading-relaxed text-muted-foreground">{rule.recommendation_text || "لا يوجد وصف مرفق."}</p>
      <div className="mt-3 flex flex-wrap gap-2 text-[11px] text-muted-foreground">
        <span>الثقة: {formatPercent(rule.confidence)}</span>
        <span>الرفع: {typeof rule.lift === "number" ? `${rule.lift.toFixed(2)}x` : "غير متوفر"}</span>
        <span>العينة: {rule.count ?? rule.antecedent_count ?? "غير متوفر"}</span>
        <span>{rule.reliability || "غير مصنف"}</span>
      </div>
    </div>
  );
}

function EmptyState({ text }: { text: string }) {
  return <div className="rounded-xl border border-dashed border-border bg-muted/20 p-4 text-sm text-muted-foreground">{text}</div>;
}
