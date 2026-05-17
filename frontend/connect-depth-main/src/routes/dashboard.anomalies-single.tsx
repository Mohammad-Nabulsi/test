import { createFileRoute } from "@tanstack/react-router";
import { motion } from "framer-motion";
import { useCallback, useEffect, useRef, useState } from "react";
import { Loader2, Upload, X } from "lucide-react";
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

function fmtPercent(value?: number): string {
  if (typeof value !== "number" || !Number.isFinite(value)) return "-";
  return `${(value * 100).toFixed(2)}%`;
}

function fmtInt(value?: number): string {
  if (typeof value !== "number" || !Number.isFinite(value)) return "-";
  return value.toLocaleString();
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

  return (
    <div className="space-y-5">
      <div className="rounded-2xl border border-border bg-card shadow-card p-5">
        <h3 className="text-sm font-semibold mb-3">Upload Dataset For Anomaly Insights</h3>

        {!loading && !result && !file ? (
          <div
            onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
            onDragLeave={() => setDragOver(false)}
            onDrop={(e) => { e.preventDefault(); setDragOver(false); handleFile(e.dataTransfer.files[0] ?? null); }}
            onClick={() => inputRef.current?.click()}
            className={`border-2 border-dashed rounded-xl p-10 text-center cursor-pointer transition-colors ${
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
          <div className="flex items-center gap-4 p-4 rounded-xl bg-accent/30">
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

        {error && <div className="mt-4 rounded-xl border border-destructive/30 bg-destructive/5 p-3 text-sm text-destructive">{error}</div>}
      </div>

      <div className="rounded-2xl border border-border bg-card shadow-card p-5">
        <h3 className="text-sm font-semibold mb-3">Sector Summary</h3>
        <div className="flex flex-wrap items-center gap-2">
          <input
            value={sectorInput}
            onChange={(e) => setSectorInput(e.target.value)}
            className="w-full md:w-80 rounded-lg border border-border bg-background px-3 py-2 text-sm"
            placeholder="Sector name"
          />
          <button
            onClick={() => runSectorSummary(sectorInput)}
            disabled={loadingSector}
            className="px-4 py-2 rounded-xl border border-border text-sm hover:bg-accent disabled:opacity-60"
          >
            {loadingSector ? "Loading sector summary..." : "Run sector summary"}
          </button>
        </div>
      </div>

      {result && (
        <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} className="rounded-2xl border border-border bg-card shadow-card p-5">
          <h3 className="text-sm font-semibold mb-3">Anomaly Insights</h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-2 mb-4">
            <div className="rounded-lg border border-border p-3"><div className="text-[11px] text-muted-foreground">Business</div><div className="font-semibold truncate">{result.business_name || "-"}</div></div>
            <div className="rounded-lg border border-border p-3"><div className="text-[11px] text-muted-foreground">Sector</div><div className="font-semibold truncate">{result.sector || "-"}</div></div>
            <div className="rounded-lg border border-border p-3"><div className="text-[11px] text-muted-foreground">Top Positive</div><div className="font-semibold">{result.top_positive_anomalies.length}</div></div>
            <div className="rounded-lg border border-border p-3"><div className="text-[11px] text-muted-foreground">Top Negative</div><div className="font-semibold">{result.top_negative_anomalies.length}</div></div>
          </div>
          <p className="text-xs text-muted-foreground mb-4">{result.message}</p>

          <div className="grid lg:grid-cols-2 gap-4">
            <section className="rounded-xl border border-emerald-300/30 bg-emerald-500/5 p-4">
              <h4 className="text-sm font-semibold mb-3">Top Positive Anomalies</h4>
              <div className="space-y-2">
                {result.top_positive_anomalies.map((post, idx) => (
                  <article key={`pos-${idx}`} className="rounded-lg border border-border bg-background p-3">
                    <div className="flex items-center justify-between gap-2 mb-2">
                      <span className="text-xs rounded-full border border-emerald-300/50 bg-emerald-500/10 px-2 py-0.5">#{idx + 1}</span>
                      <span className="text-xs font-medium">{post.post_type ?? "-"}</span>
                    </div>
                    <p className="text-sm line-clamp-3 mb-2">{post.caption_text || "-"}</p>
                    <div className="grid grid-cols-2 gap-x-3 gap-y-1 text-xs text-muted-foreground">
                      <div>Engagement: <span className="font-medium text-foreground">{fmtPercent(post.engagement_rate)}</span></div>
                      <div>Views: <span className="font-medium text-foreground">{fmtInt(post.views_count)}</span></div>
                      <div>Likes: <span className="font-medium text-foreground">{fmtInt(post.likes_count)}</span></div>
                      <div>Comments: <span className="font-medium text-foreground">{fmtInt(post.comments_count)}</span></div>
                    </div>
                  </article>
                ))}
                {result.top_positive_anomalies.length === 0 && (
                  <p className="text-xs text-muted-foreground">No positive anomalies.</p>
                )}
              </div>
            </section>

            <section className="rounded-xl border border-destructive/30 bg-destructive/5 p-4">
              <h4 className="text-sm font-semibold mb-3">Top Negative Anomalies</h4>
              <div className="space-y-2">
                {result.top_negative_anomalies.map((post, idx) => (
                  <article key={`neg-${idx}`} className="rounded-lg border border-border bg-background p-3">
                    <div className="flex items-center justify-between gap-2 mb-2">
                      <span className="text-xs rounded-full border border-destructive/40 bg-destructive/10 px-2 py-0.5">#{idx + 1}</span>
                      <span className="text-xs font-medium">{post.post_type ?? "-"}</span>
                    </div>
                    <p className="text-sm line-clamp-3 mb-2">{post.caption_text || "-"}</p>
                    <div className="grid grid-cols-2 gap-x-3 gap-y-1 text-xs text-muted-foreground">
                      <div>Engagement: <span className="font-medium text-foreground">{fmtPercent(post.engagement_rate)}</span></div>
                      <div>Views: <span className="font-medium text-foreground">{fmtInt(post.views_count)}</span></div>
                      <div>Likes: <span className="font-medium text-foreground">{fmtInt(post.likes_count)}</span></div>
                      <div>Comments: <span className="font-medium text-foreground">{fmtInt(post.comments_count)}</span></div>
                    </div>
                  </article>
                ))}
                {result.top_negative_anomalies.length === 0 && (
                  <p className="text-xs text-muted-foreground">No negative anomalies.</p>
                )}
              </div>
            </section>
          </div>
        </motion.div>
      )}

      {sectorSummary && (
        <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} className="rounded-2xl border border-border bg-card shadow-card p-5">
          <h3 className="text-sm font-semibold mb-3">Sector Summary Response</h3>
          {sectorSummary.length === 0 ? (
            <p className="text-sm text-muted-foreground">No sector summary records returned.</p>
          ) : (
            <pre className="max-h-80 overflow-auto rounded-xl border border-border bg-background p-3 text-xs">
              {JSON.stringify(sectorSummary, null, 2)}
            </pre>
          )}
        </motion.div>
      )}
    </div>
  );
}
