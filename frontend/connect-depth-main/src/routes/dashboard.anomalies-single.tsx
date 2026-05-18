import { createFileRoute } from "@tanstack/react-router";
import { motion } from "framer-motion";
import { useCallback, useEffect, useRef, useState } from "react";
import { AlertCircle, BarChart3, Download, FileText, Loader2, TrendingDown, TrendingUp, Upload, X } from "lucide-react";
import { getPendingDatasetFile } from "@/lib/upload-state";

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000";

export const Route = createFileRoute("/dashboard/anomalies-single")({
  component: AnomaliesSinglePage,
});

type UploadResponse = { dataset_id: string };

type AnomalyPost = {
  business_name?: string;
  sector?: string;
  post_date?: string | null;
  post_type?: string;
  caption_text?: string;
  engagement_rate?: number;
  views_count?: number;
  likes_count?: number;
  comments_count?: number;
  caption_length?: number;
  hashtags_count?: number;
  emoji_count?: number;
  promo_post?: boolean;
  anomaly_type?: string;
  best_method?: string;
  best_setting?: string;
};

type AnomaliesResult = {
  business_name: string;
  sector: string;
  message: string;
  top_positive_anomalies: AnomalyPost[];
  top_negative_anomalies: AnomalyPost[];
  recommendations: unknown[];
  sector_anomaly_summary: unknown[];
  chart_url: string | null;
  csv_outputs: string[];
};

function resolveAssetUrl(path?: string | null): string {
  if (!path) return "";
  if (path.startsWith("http://") || path.startsWith("https://")) return path;
  return `${API_BASE.replace(/\/$/, "")}/${path.replace(/^\/+/, "")}`;
}

function fmtPercent(value?: number | null): string {
  if (typeof value !== "number" || !Number.isFinite(value)) return "-";
  return `${(value * 100).toFixed(2)}%`;
}

function fmtInt(value?: number | null): string {
  if (typeof value !== "number" || !Number.isFinite(value)) return "-";
  return value.toLocaleString();
}

function fmtValue(value: unknown): string {
  if (value === null || value === undefined || value === "") return "Not available";
  if (typeof value === "number") return Number.isInteger(value) ? fmtInt(value) : value.toFixed(4);
  if (typeof value === "boolean") return value ? "Yes" : "No";
  return String(value);
}

function EmptyPanel({ title, message }: { title: string; message: string }) {
  return (
    <div className="rounded-2xl border border-dashed border-border bg-muted/20 p-6 text-center">
      <FileText className="mx-auto h-7 w-7 text-muted-foreground" />
      <h4 className="mt-3 text-sm font-semibold">{title}</h4>
      <p className="mt-1 text-sm text-muted-foreground">{message}</p>
    </div>
  );
}

function MetricCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-2xl border border-border/80 bg-background p-4 shadow-card">
      <div className="text-xs text-muted-foreground">{label}</div>
      <div className="mt-1 text-lg font-semibold num-display truncate">{value}</div>
    </div>
  );
}

function DetailGrid({ post }: { post: AnomalyPost }) {
  const items = [
    ["Post Type", post.post_type],
    ["Post Date", post.post_date ?? "Not available"],
    ["Engagement Rate", fmtPercent(post.engagement_rate)],
    ["Views", fmtInt(post.views_count)],
    ["Likes", fmtInt(post.likes_count)],
    ["Comments", fmtInt(post.comments_count)],
    ["Caption Length", fmtInt(post.caption_length)],
    ["Hashtags", fmtInt(post.hashtags_count)],
    ["Emojis", fmtInt(post.emoji_count)],
    ["Promo Post", post.promo_post === undefined ? "Not available" : post.promo_post ? "Yes" : "No"],
    ["Anomaly Type", post.anomaly_type],
  ];

  return (
    <dl className="grid grid-cols-2 md:grid-cols-3 gap-2 text-xs">
      {items.map(([label, value]) => (
        <div key={label} className="rounded-lg border border-border/70 bg-muted/15 p-2.5">
          <dt className="text-muted-foreground">{label}</dt>
          <dd className="mt-1 font-medium text-foreground break-words">{fmtValue(value)}</dd>
        </div>
      ))}
    </dl>
  );
}

function AnomalyList({ title, tone, items }: { title: string; tone: "positive" | "negative"; items: AnomalyPost[] }) {
  const positive = tone === "positive";

  return (
    <section className={`rounded-2xl border p-4 shadow-card ${positive ? "border-emerald-300/40 bg-emerald-500/5" : "border-rose-300/40 bg-rose-500/5"}`}>
      <div className="mb-4 flex items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          {positive ? <TrendingUp className="h-4 w-4 text-emerald-600" /> : <TrendingDown className="h-4 w-4 text-rose-600" />}
          <h4 className="text-sm font-semibold">{title}</h4>
        </div>
        <span className={`rounded-full border px-2.5 py-1 text-xs font-medium ${positive ? "border-emerald-300/50 bg-emerald-500/10 text-emerald-700" : "border-rose-300/50 bg-rose-500/10 text-rose-700"}`}>
          {items.length} posts
        </span>
      </div>

      {items.length === 0 ? (
        <EmptyPanel title={`No ${positive ? "positive" : "negative"} anomalies`} message="The backend did not return records for this group." />
      ) : (
        <div className="space-y-3">
          {items.map((post, idx) => (
            <article key={`${tone}-${idx}`} className="rounded-xl border border-border/80 bg-card p-4">
              <div className="mb-3 flex items-start justify-between gap-3">
                <div>
                  <div className="text-xs text-muted-foreground">Anomaly #{idx + 1}</div>
                  <h5 className="mt-1 text-sm font-semibold">{fmtValue(post.post_type)}</h5>
                </div>
                <span className={`rounded-full border px-2.5 py-1 text-xs ${positive ? "border-emerald-300/50 bg-emerald-500/10 text-emerald-700" : "border-rose-300/50 bg-rose-500/10 text-rose-700"}`}>
                  {fmtValue(post.anomaly_type)}
                </span>
              </div>
              <p className="mb-3 rounded-lg border border-border/70 bg-muted/15 p-3 text-sm leading-6">
                {post.caption_text || "No caption available."}
              </p>
              <DetailGrid post={post} />
            </article>
          ))}
        </div>
      )}
    </section>
  );
}

function UnknownList({ title, items, emptyMessage }: { title: string; items: unknown[] | null | undefined; emptyMessage: string }) {
  if (!Array.isArray(items) || items.length === 0) {
    return <EmptyPanel title={title} message={emptyMessage} />;
  }

  return (
    <section className="rounded-2xl border border-border/80 bg-card p-4 shadow-card">
      <h4 className="mb-3 text-sm font-semibold">{title}</h4>
      <div className="space-y-2">
        {items.map((item, idx) => {
          if (item && typeof item === "object" && !Array.isArray(item)) {
            return (
              <div key={idx} className="rounded-xl border border-border/70 bg-background p-3">
                <div className="mb-2 text-xs font-medium text-muted-foreground">Item {idx + 1}</div>
                <dl className="grid grid-cols-1 md:grid-cols-2 gap-2 text-xs">
                  {Object.entries(item as Record<string, unknown>).map(([key, value]) => (
                    <div key={key} className="rounded-lg bg-muted/20 p-2">
                      <dt className="text-muted-foreground">{key.replaceAll("_", " ")}</dt>
                      <dd className="mt-1 font-medium break-words">{fmtValue(value)}</dd>
                    </div>
                  ))}
                </dl>
              </div>
            );
          }

          return (
            <div key={idx} className="rounded-xl border border-border/70 bg-background p-3 text-sm">
              {fmtValue(item)}
            </div>
          );
        })}
      </div>
    </section>
  );
}

function CsvOutputs({ outputs }: { outputs: string[] | null | undefined }) {
  if (!Array.isArray(outputs) || outputs.length === 0) {
    return <EmptyPanel title="CSV Outputs" message="No CSV outputs available." />;
  }

  return (
    <section className="rounded-2xl border border-border/80 bg-card p-4 shadow-card">
      <h4 className="mb-3 text-sm font-semibold">CSV Outputs</h4>
      <div className="space-y-2">
        {outputs.map((path) => (
          <a
            key={path}
            href={resolveAssetUrl(path)}
            target="_blank"
            rel="noreferrer"
            className="flex items-center gap-2 rounded-xl border border-border/70 bg-background p-3 text-sm hover:bg-muted/25"
          >
            <Download className="h-4 w-4 text-[var(--brand)]" />
            <span className="font-mono break-all">{path}</span>
          </a>
        ))}
      </div>
    </section>
  );
}

function AnomaliesSinglePage() {
  const [file, setFile] = useState<File | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const [loading, setLoading] = useState(false);
  const [loadingSector, setLoadingSector] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [uploadedDatasetPath, setUploadedDatasetPath] = useState<string | null>(null);
  const [result, setResult] = useState<AnomaliesResult | null>(null);
  const [sectorSummary, setSectorSummary] = useState<unknown[] | null>(null);
  const [sectorInput, setSectorInput] = useState("restaurants");
  const inputRef = useRef<HTMLInputElement>(null);

  const runSectorSummary = useCallback(async (sector: string) => {
    const normalized = sector.trim();
    if (!normalized) return;
    setLoadingSector(true);
    setError(null);
    try {
      const res = await fetch(
        `${API_BASE}/api/anomalies/sector-summary-single?sector=${encodeURIComponent(normalized)}`,
      );
      if (!res.ok) {
        const text = await res.text();
        throw new Error(text || `Sector summary failed (${res.status})`);
      }
      setSectorSummary((await res.json()) as unknown[]);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Sector summary failed");
    } finally {
      setLoadingSector(false);
    }
  }, []);

  const runAnomaliesForFile = useCallback(async (uploadFile: File) => {
    setLoading(true);
    setError(null);
    setResult(null);
    setSectorSummary(null);
    try {
      const form = new FormData();
      form.append("file", uploadFile);

      const uploadRes = await fetch(`${API_BASE}/api/upload`, {
        method: "POST",
        body: form,
      });
      if (!uploadRes.ok) {
        const text = await uploadRes.text();
        throw new Error(text || `Upload failed (${uploadRes.status})`);
      }

      const uploadData = (await uploadRes.json()) as UploadResponse;
      if (!uploadData.dataset_id) {
        throw new Error("Upload response missing dataset_id");
      }

      const path = `storage/raw/${uploadData.dataset_id}/raw${uploadFile.name.toLowerCase().endsWith(".json") ? ".json" : ".csv"}`;
      setUploadedDatasetPath(path);

      const anomaliesRes = await fetch(`${API_BASE}/api/anomalies/analyze-single`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ uploaded_file_path: path }),
      });

      if (!anomaliesRes.ok) {
        const text = await anomaliesRes.text();
        throw new Error(text || `Anomaly analysis failed (${anomaliesRes.status})`);
      }

      const data = (await anomaliesRes.json()) as AnomaliesResult;
      setResult(data);

      const sector = data.sector?.trim() || sectorInput;
      if (sector) {
        setSectorInput(sector);
        await runSectorSummary(sector);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "Anomaly analysis failed");
    } finally {
      setLoading(false);
    }
  }, [runSectorSummary, sectorInput]);

  useEffect(() => {
    const pending = getPendingDatasetFile();
    if (pending) {
      setFile(pending);
      runAnomaliesForFile(pending);
    }
  }, [runAnomaliesForFile]);

  const handleFile = useCallback((f: File | null) => {
    if (!f) return;
    const name = f.name.toLowerCase();
    if (!name.endsWith(".json") && !name.endsWith(".csv")) {
      setError("Please upload JSON or CSV only");
      return;
    }
    setFile(f);
    setError(null);
    runAnomaliesForFile(f);
  }, [runAnomaliesForFile]);

  const clearFile = () => {
    setFile(null);
    setResult(null);
    setSectorSummary(null);
    setUploadedDatasetPath(null);
    setError(null);
    if (inputRef.current) inputRef.current.value = "";
  };

  const chartSrc = resolveAssetUrl(result?.chart_url);

  return (
    <div className="space-y-5">
      <section className="rounded-3xl border border-border bg-card p-5 shadow-card">
        <div className="mb-4 flex items-start justify-between gap-3 flex-wrap">
          <div>
            <h2 className="text-xl font-semibold font-display">Anomaly Detection</h2>
            <p className="mt-1 text-sm text-muted-foreground">Upload a dataset to surface unusually strong and weak posts.</p>
          </div>
          {result ? (
            <span className="rounded-full border border-emerald-300/50 bg-emerald-500/10 px-3 py-1 text-xs font-medium text-emerald-700">
              Analysis complete
            </span>
          ) : null}
        </div>

        {!loading && !result && !file ? (
          <div
            onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
            onDragLeave={() => setDragOver(false)}
            onDrop={(e) => { e.preventDefault(); setDragOver(false); handleFile(e.dataTransfer.files[0] ?? null); }}
            onClick={() => inputRef.current?.click()}
            className={`border-2 border-dashed rounded-2xl p-10 text-center cursor-pointer transition-colors ${
              dragOver ? "border-[var(--brand)] bg-accent/30" : "border-border hover:border-[var(--brand)]/50 hover:bg-accent/10"
            }`}
          >
            <Upload className="h-8 w-8 mx-auto mb-3 text-muted-foreground" />
            <p className="text-sm font-medium">Drop JSON or CSV here</p>
            <p className="text-xs text-muted-foreground mt-1">Or click to browse</p>
            <input
              ref={inputRef}
              type="file"
              accept=".json,.csv"
              className="hidden"
              onChange={(e) => handleFile(e.target.files?.[0] ?? null)}
            />
          </div>
        ) : file ? (
          <div className="flex items-center gap-4 p-4 rounded-2xl bg-accent/30">
            <div className="flex-1 min-w-0">
              <p className="text-sm font-medium truncate">{file.name}</p>
              <p className="text-xs text-muted-foreground">{(file.size / 1024).toFixed(1)} KB</p>
            </div>
            <button onClick={clearFile} className="p-2 rounded-lg hover:bg-muted transition-colors" disabled={loading}>
              <X className="h-4 w-4" />
            </button>
            {loading && (
              <span className="px-4 py-2 rounded-xl bg-gradient-brand text-white text-sm font-medium flex items-center gap-2">
                <Loader2 className="h-3.5 w-3.5 animate-spin" />
                Analyzing...
              </span>
            )}
          </div>
        ) : null}

        {uploadedDatasetPath && (
          <p className="text-xs text-muted-foreground mt-3">
            Uploaded path: <span className="font-mono">{uploadedDatasetPath}</span>
          </p>
        )}

        {error && (
          <div className="mt-4 rounded-xl border border-destructive/30 bg-destructive/5 p-3 text-sm text-destructive flex items-start gap-2">
            <AlertCircle className="mt-0.5 h-4 w-4" />
            <span>{error}</span>
          </div>
        )}
      </section>

      <section className="rounded-3xl border border-border bg-card p-5 shadow-card">
        <h3 className="text-sm font-semibold mb-3">Sector Summary</h3>
        <div className="flex flex-wrap items-center gap-2">
          <input
            value={sectorInput}
            onChange={(e) => setSectorInput(e.target.value)}
            className="w-full md:w-80 rounded-xl border border-border bg-background px-3 py-2 text-sm"
            placeholder="Sector name"
          />
          <button
            onClick={() => runSectorSummary(sectorInput)}
            disabled={loadingSector}
            className="px-4 py-2 rounded-xl border border-border text-sm hover:bg-accent disabled:opacity-60 inline-flex items-center gap-2"
          >
            {loadingSector ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <BarChart3 className="h-3.5 w-3.5" />}
            {loadingSector ? "Loading sector summary..." : "Run sector summary"}
          </button>
        </div>
      </section>

      {!loading && !result ? (
        <EmptyPanel title="No anomaly results yet" message="Upload a dataset to generate anomaly insights." />
      ) : null}

      {result && (
        <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} className="space-y-5">
          <section className="rounded-3xl border border-border bg-card p-5 shadow-card">
            <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-3">
              <MetricCard label="Business" value={result.business_name || "-"} />
              <MetricCard label="Sector" value={result.sector || "-"} />
              <MetricCard label="Positive Anomalies" value={fmtInt(result.top_positive_anomalies?.length ?? 0)} />
              <MetricCard label="Negative Anomalies" value={fmtInt(result.top_negative_anomalies?.length ?? 0)} />
            </div>
            <div className="mt-4 rounded-2xl border border-border/80 bg-muted/20 p-4">
              <div className="text-xs text-muted-foreground">Message</div>
              <p className="mt-1 text-sm leading-6">{result.message || "Analysis completed."}</p>
            </div>
          </section>

          <div className="grid grid-cols-1 xl:grid-cols-2 gap-5">
            <AnomalyList title="Top Positive Anomalies" tone="positive" items={result.top_positive_anomalies ?? []} />
            <AnomalyList title="Top Negative Anomalies" tone="negative" items={result.top_negative_anomalies ?? []} />
          </div>

          <div className="grid grid-cols-1 xl:grid-cols-2 gap-5">
            <UnknownList title="Recommendations" items={result.recommendations} emptyMessage="No recommendations returned yet." />
            <UnknownList title="Sector Anomaly Summary" items={result.sector_anomaly_summary} emptyMessage="No sector anomaly summary returned yet." />
          </div>

          <section className="rounded-2xl border border-border/80 bg-card p-4 shadow-card">
            <h4 className="mb-3 text-sm font-semibold">Anomaly Chart</h4>
            {chartSrc ? (
              <img src={chartSrc} alt="Anomaly detection chart" className="w-full max-h-[460px] rounded-xl border border-border/70 bg-muted/20 object-contain" />
            ) : (
              <EmptyPanel title="No chart generated yet." message="Run analysis again after the backend produces a chart URL." />
            )}
          </section>

          <CsvOutputs outputs={result.csv_outputs} />
        </motion.div>
      )}

      {sectorSummary ? (
        <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }}>
          <UnknownList title="Sector Summary" items={sectorSummary} emptyMessage="No sector summary records returned." />
        </motion.div>
      ) : null}
    </div>
  );
}
