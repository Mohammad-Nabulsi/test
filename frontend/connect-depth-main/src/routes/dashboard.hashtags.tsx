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

type ApiCallState = {
  ok: boolean;
  data?: unknown;
  error?: string;
};

type RunState = {
  datasetId: string;
  hashtagFromUpload: ApiCallState;
  getHashtag: ApiCallState;
};

type StaticTopicCardCounts = {
  topics_total?: number;
  posts_with_topics?: number;
  recommendations_total?: number;
  terms_total?: number;
};

type StaticTopicPreview = {
  topic_id: string;
  terms: string[];
};

type StaticTopicResponse = {
  cards?: StaticTopicCardCounts;
  recommendations?: Array<Record<string, unknown>>;
  topic_summary?: Array<Record<string, unknown>>;
  topics_preview?: StaticTopicPreview[];
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

function parseSimpleCsv(text: string): Array<Record<string, string>> {
  const lines = text.split(/\r?\n/).filter(Boolean);
  if (!lines.length) return [];
  const headers = lines[0].split(",").map((h) => h.trim());
  return lines.slice(1).map((line) => {
    const cols = line.split(",");
    const row: Record<string, string> = {};
    headers.forEach((h, i) => {
      row[h] = (cols[i] ?? "").trim();
    });
    return row;
  });
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
  const [staticData, setStaticData] = useState<StaticTopicResponse | null>(null);
  const [staticError, setStaticError] = useState<string | null>(null);

  useEffect(() => {
    const latest = getLatestCaptionAnalysis();
    setAnalysis(latest);
    if (latest?.dataset_id) setDatasetIdInput(latest.dataset_id);
  }, []);

  useEffect(() => {
    let active = true;
    (async () => {
      try {
        const [topicsRes, summaryRes, recsRes, datasetRes, termsRes] = await Promise.all([
          fetch("/static-topic-content/topics.json"),
          fetch("/static-topic-content/topic_summary.csv"),
          fetch("/static-topic-content/dynamic_recommendations.csv"),
          fetch("/static-topic-content/dataset_with_topics.csv"),
          fetch("/static-topic-content/topic_terms_exact_1_2_3grams.csv"),
        ]);
        if (!active) return;
        if (!topicsRes.ok || !summaryRes.ok || !recsRes.ok || !datasetRes.ok || !termsRes.ok) {
          setStaticError("تعذر تحميل الملفات الثابتة من الواجهة.");
          return;
        }

        const topicsJson = (await topicsRes.json()) as { topic_representations?: Record<string, Array<[string, number]>> };
        const summaryRows = parseSimpleCsv(await summaryRes.text());
        const recRows = parseSimpleCsv(await recsRes.text());
        const datasetRows = parseSimpleCsv(await datasetRes.text());
        const termRows = parseSimpleCsv(await termsRes.text());

        const topicsPreview: StaticTopicPreview[] = Object.entries(topicsJson.topic_representations ?? {})
          .slice(0, 8)
          .map(([topicId, terms]) => ({
            topic_id: topicId,
            terms: (terms ?? []).slice(0, 8).map((item) => String(item[0] ?? "")).filter(Boolean),
          }));

        setStaticData({
          cards: {
            topics_total: summaryRows.length,
            posts_with_topics: datasetRows.length,
            recommendations_total: recRows.length,
            terms_total: termRows.length,
          },
          topic_summary: summaryRows.slice(0, 8),
          recommendations: recRows.slice(0, 8),
          topics_preview: topicsPreview,
        });
        setStaticError(null);
      } catch {
        if (!active) return;
        setStaticError("تعذر تحميل الملفات الثابتة من الواجهة.");
      }
    })();
    return () => {
      active = false;
    };
  }, []);

  const hashtagData =
    (runState?.getHashtag.data as HashtagResponse | undefined) ??
    (analysis?.hashtag_stage as HashtagResponse | undefined);

  const useTags = useMemo(() => (hashtagData?.top_recommendations ?? []).slice(0, 10), [hashtagData]);
  const avoidTags = useMemo(() => (hashtagData?.warning_recommendations ?? []).slice(0, 8), [hashtagData]);
  const hasData = Boolean(hashtagData);
  const adaptiveMinCount = metricValue(hashtagData?.summary, "adaptive_min_count");
  const activeDatasetId = runState?.datasetId || datasetIdInput || analysis?.dataset_id || "";

  async function runHashtagFromUpload(selectedFile: File) {
    setRunning(true);
    setRunError(null);
    setRunState(null);

    const uploadForm = new FormData();
    uploadForm.append("file", selectedFile);

    const hashtagFromUpload = await callApi(`${API_BASE}/api/datasets/stages/hashtag`, {
      method: "POST",
      body: uploadForm,
    });

    if (!hashtagFromUpload.ok) {
      setRunning(false);
      setRunError(hashtagFromUpload.error || "فشل تشغيل مرحلة الهاشتاقات.");
      return;
    }

    const datasetId = String((hashtagFromUpload.data as { dataset_id?: string })?.dataset_id || "");
    if (!datasetId) {
      setRunning(false);
      setRunError("لم يتم العثور على dataset_id في الاستجابة.");
      return;
    }

    setDatasetIdInput(datasetId);
    const getHashtag = await callApi(`${API_BASE}/api/datasets/${datasetId}/hashtag-recommendations`);

    setRunState({ datasetId, hashtagFromUpload, getHashtag });
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
    const getHashtag = await callApi(`${API_BASE}/api/datasets/${datasetId}/hashtag-recommendations`);
    setRunState({
      datasetId,
      hashtagFromUpload: { ok: true, data: { note: "لم يتم التشغيل في وضع التحديث" } },
      getHashtag,
    });
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
          <h2 className="font-display text-2xl tracking-tight md:text-3xl">ارفع الملف واحصل على توصيات الهاشتاقات</h2>
          <p className="max-w-2xl text-sm leading-relaxed text-muted-foreground">
            هذه الصفحة تشغّل واجهات توصيات الهاشتاق فقط، بدون استدعاءات BERTopic.
          </p>
        </div>
      </motion.div>

      <section className="rounded-2xl border border-border bg-card p-5 shadow-card space-y-4">
        <div className="flex flex-col gap-3 lg:flex-row lg:items-end">
          <div className="flex-1">
            <label className="mb-2 block text-xs text-muted-foreground">ارفع ملف .json أو .csv لتشغيل ربط الهاشتاقات</label>
            <input
              type="file"
              accept=".csv,.json"
              onChange={(event) => setFile(event.target.files?.[0] ?? null)}
              className="block w-full rounded-xl border border-border bg-background px-3 py-2 text-sm"
            />
          </div>
          <button
            disabled={!file || running}
            onClick={() => file && runHashtagFromUpload(file)}
            className="inline-flex items-center gap-2 rounded-xl bg-[var(--brand)] px-4 py-2 text-sm font-medium text-white disabled:opacity-50"
          >
            <UploadCloud className="h-4 w-4" />
            {running ? "جاري التشغيل..." : "تشغيل توصيات الهاشتاقات"}
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
            {running ? "جاري التحديث..." : "جلب النتائج"}
          </button>
        </div>

        {runError && <div className="rounded-xl border border-destructive/30 bg-destructive/5 p-3 text-xs text-destructive">{runError}</div>}
        <div className="text-xs text-muted-foreground">
          dataset_id النشط: <span className="font-medium text-foreground">{activeDatasetId || "غير متوفر"}</span>
        </div>
      </section>

      {!hasData && (
        <div className="rounded-2xl border border-border bg-card p-6 text-sm text-muted-foreground shadow-card">
          ارفع ملفًا لعرض بطاقات توصيات الهاشتاقات.
        </div>
      )}

      {hasData && (
        <>
          <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
            <SummaryCard icon={Hash} label="هاشتاقات مقترحة" value={useTags.length} hint={`أدنى حد تكيفي: ${adaptiveMinCount ?? "غير متوفر"}`} />
            <SummaryCard icon={AlertTriangle} label="هاشتاقات تحتاج حذرًا" value={avoidTags.length} hint={hashtagData?.used_fallback ? "تم استخدام بديل احتياطي" : "مستخرجة من قواعد الارتباط"} />
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
          <h3 className="text-base font-semibold">Static Content</h3>
          <span className="rounded-full border border-border bg-muted px-2.5 py-1 text-xs text-muted-foreground">ثابت</span>
        </div>

        {staticError && <div className="rounded-xl border border-destructive/30 bg-destructive/5 p-3 text-xs text-destructive">{staticError}</div>}

        {!staticError && staticData?.cards && (
          <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
            <SummaryCard icon={Hash} label="إجمالي المواضيع" value={staticData.cards.topics_total ?? 0} hint="من topics.json" />
            <SummaryCard icon={CheckCircle2} label="منشورات موسومة" value={staticData.cards.posts_with_topics ?? 0} hint="dataset_with_topics.csv" />
            <SummaryCard icon={Sparkles} label="توصيات ثابتة" value={staticData.cards.recommendations_total ?? 0} hint="dynamic_recommendations.csv" />
            <SummaryCard icon={AlertTriangle} label="مصطلحات ثابتة" value={staticData.cards.terms_total ?? 0} hint="topic_terms_exact_1_2_3grams.csv" />
          </div>
        )}

        {!!staticData?.topics_preview?.length && (
          <div className="rounded-xl border border-border p-4">
            <h4 className="mb-3 text-sm font-semibold">مواضيع BERTopic (ثابتة)</h4>
            <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
              {staticData.topics_preview.map((item) => (
                <div key={item.topic_id} className="rounded-lg border border-border bg-muted/20 p-3">
                  <div className="mb-2 text-xs text-muted-foreground">Topic {item.topic_id}</div>
                  <div className="flex flex-wrap gap-1.5">
                    {item.terms.map((term) => (
                      <span key={`${item.topic_id}-${term}`} className="rounded-full bg-card px-2 py-1 text-xs">
                        {term}
                      </span>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {!!staticData?.recommendations?.length && (
          <div className="rounded-xl border border-border p-4">
            <h4 className="mb-3 text-sm font-semibold">توصيات ثابتة</h4>
            <div className="space-y-2">
              {staticData.recommendations.slice(0, 6).map((row, idx) => (
                <div key={`static-rec-${idx}`} className="rounded-lg border border-border bg-muted/20 p-3 text-xs">
                  <div className="font-medium">{String(row.action ?? "Action")}</div>
                  <div className="mt-1 text-muted-foreground">{String(row.reason ?? "")}</div>
                </div>
              ))}
            </div>
          </div>
        )}

        <div className="rounded-xl border border-border p-4">
          <h4 className="mb-3 text-sm font-semibold">خريطة المسافات بين المواضيع (ثابتة)</h4>
          <iframe
            src="/static-topic-content/intertopic_distance_map.html"
            title="Static Intertopic Map"
            className="h-[500px] w-full rounded-lg border border-border bg-background"
            loading="lazy"
          />
        </div>
      </section>
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
