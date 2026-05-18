import { createFileRoute } from "@tanstack/react-router";
import { motion } from "framer-motion";
import { useCallback, useEffect, useRef, useState } from "react";
import { Loader2, Upload, X } from "lucide-react";
import { getPendingDatasetFile } from "@/lib/upload-state";

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000";

export const Route = createFileRoute("/dashboard/business-momentum-single")({
  component: BusinessMomentumSinglePage,
});

type UploadResponse = { dataset_id: string };

type MomentumResult = {
  business_name: string;
  sector: string;
  trend: string;
  comparison_label: string;
  latest_business_engagement_rate: number;
  latest_sector_engagement_rate: number | null;
  difference_from_sector: number;
  message: string;
  chart_url: string;
  csv_outputs: string[];
};

type SectorSummaryItem = {
  sector: string;
  avg_engagement_rate: number;
  trend: string;
};

function fmtPercent(value?: number | null): string {
  if (typeof value !== "number" || !Number.isFinite(value)) return "-";
  return `${(value * 100).toFixed(2)}%`;
}

function trendTone(trend: string): string {
  const t = trend.toLowerCase();
  if (t === "improving") return "border-emerald-300/50 bg-emerald-500/10 text-emerald-700";
  if (t === "declining") return "border-destructive/40 bg-destructive/10 text-destructive";
  return "border-amber-300/50 bg-amber-500/10 text-amber-700";
}

function BusinessMomentumSinglePage() {
  const [file, setFile] = useState<File | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const [loadingAnalyze, setLoadingAnalyze] = useState(false);
  const [loadingStatus, setLoadingStatus] = useState(false);
  const [loadingSector, setLoadingSector] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [uploadedDatasetPath, setUploadedDatasetPath] = useState<string | null>(null);
  const [analyzeResult, setAnalyzeResult] = useState<MomentumResult | null>(null);
  const [statusResult, setStatusResult] = useState<MomentumResult | null>(null);
  const [sectorSummary, setSectorSummary] = useState<SectorSummaryItem[] | null>(null);
  const [businessNameInput, setBusinessNameInput] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);

  const runStatusSingle = useCallback(async (businessName: string) => {
    const normalized = businessName.trim();
    if (!normalized) return;
    setLoadingStatus(true);
    try {
      const res = await fetch(
        `${API_BASE}/api/business-momentum/status-single?business_name=${encodeURIComponent(normalized)}`,
      );
      if (!res.ok) {
        const text = await res.text();
        throw new Error(text || `Status request failed (${res.status})`);
      }
      setStatusResult((await res.json()) as MomentumResult);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Status request failed");
    } finally {
      setLoadingStatus(false);
    }
  }, []);

  const runSectorSummary = useCallback(async () => {
    setLoadingSector(true);
    try {
      const res = await fetch(`${API_BASE}/api/business-momentum/sector-summary-single`);
      if (!res.ok) {
        const text = await res.text();
        throw new Error(text || `Sector summary failed (${res.status})`);
      }
      setSectorSummary((await res.json()) as SectorSummaryItem[]);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Sector summary failed");
    } finally {
      setLoadingSector(false);
    }
  }, []);

  const runAnalyzeSingleForFile = useCallback(async (uploadFile: File) => {
    setLoadingAnalyze(true);
    setError(null);
    setAnalyzeResult(null);
    setStatusResult(null);
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

      const res = await fetch(`${API_BASE}/api/business-momentum/analyze-single`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ uploaded_file_path: path }),
      });
      if (!res.ok) {
        const text = await res.text();
        throw new Error(text || `Analyze failed (${res.status})`);
      }

      const data = (await res.json()) as MomentumResult;
      setAnalyzeResult(data);
      setBusinessNameInput(data.business_name || "");

      if (data.business_name) {
        await runStatusSingle(data.business_name);
      }
      await runSectorSummary();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Analyze failed");
    } finally {
      setLoadingAnalyze(false);
    }
  }, [runSectorSummary, runStatusSingle]);

  useEffect(() => {
    const pending = getPendingDatasetFile();
    if (pending) {
      setFile(pending);
      runAnalyzeSingleForFile(pending);
    }
  }, [runAnalyzeSingleForFile]);

  const handleFile = useCallback((f: File | null) => {
    if (!f) return;
    const name = f.name.toLowerCase();
    if (!name.endsWith(".json") && !name.endsWith(".csv")) {
      setError("Please upload JSON or CSV only");
      return;
    }
    setFile(f);
    setError(null);
    runAnalyzeSingleForFile(f);
  }, [runAnalyzeSingleForFile]);

  const clearFile = () => {
    setFile(null);
    setAnalyzeResult(null);
    setStatusResult(null);
    setSectorSummary(null);
    setUploadedDatasetPath(null);
    setBusinessNameInput("");
    setError(null);
    if (inputRef.current) inputRef.current.value = "";
  };

  return (
    <div className="space-y-5">
      <div className="rounded-2xl border border-border bg-card shadow-card p-5">
        <h3 className="text-sm font-semibold mb-3">Upload Dataset For Business Momentum</h3>

        {!loadingAnalyze && !analyzeResult && !file ? (
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
            <button onClick={clearFile} className="p-2 rounded-lg hover:bg-muted transition-colors" disabled={loadingAnalyze}>
              <X className="h-4 w-4" />
            </button>
            {loadingAnalyze && (
              <span className="px-4 py-2 rounded-xl bg-gradient-brand text-white text-sm font-medium flex items-center gap-2">
                <Loader2 className="h-3.5 w-3.5 animate-spin" />
                Running analyze...
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
        <h3 className="text-sm font-semibold mb-3">Manual Endpoint Triggers</h3>
        <div className="flex flex-wrap items-center gap-2">
          <input
            value={businessNameInput}
            onChange={(e) => setBusinessNameInput(e.target.value)}
            className="w-full md:w-80 rounded-lg border border-border bg-background px-3 py-2 text-sm"
            placeholder="Business name for status-single"
          />
          <button
            onClick={() => runStatusSingle(businessNameInput)}
            disabled={loadingStatus}
            className="px-4 py-2 rounded-xl border border-border text-sm hover:bg-accent disabled:opacity-60"
          >
            {loadingStatus ? "Running status..." : "Run status-single"}
          </button>
          <button
            onClick={runSectorSummary}
            disabled={loadingSector}
            className="px-4 py-2 rounded-xl border border-border text-sm hover:bg-accent disabled:opacity-60"
          >
            {loadingSector ? "Loading sectors..." : "Run sector-summary-single"}
          </button>
        </div>
      </div>

      {analyzeResult && (
        <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} className="rounded-2xl border border-border bg-card shadow-card p-5">
          <h3 className="text-sm font-semibold mb-3">Analyze Single Insights</h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-2 mb-4">
            <div className="rounded-lg border border-border p-3"><div className="text-[11px] text-muted-foreground">Business</div><div className="font-semibold truncate">{analyzeResult.business_name || "-"}</div></div>
            <div className="rounded-lg border border-border p-3"><div className="text-[11px] text-muted-foreground">Sector</div><div className="font-semibold truncate">{analyzeResult.sector || "-"}</div></div>
            <div className="rounded-lg border border-border p-3"><div className="text-[11px] text-muted-foreground">Trend</div><div><span className={`inline-flex rounded-full px-2 py-0.5 text-xs border ${trendTone(analyzeResult.trend)}`}>{analyzeResult.trend}</span></div></div>
            <div className="rounded-lg border border-border p-3"><div className="text-[11px] text-muted-foreground">Comparison</div><div className="font-semibold truncate">{analyzeResult.comparison_label}</div></div>
          </div>
          <div className="grid md:grid-cols-3 gap-2 mb-4">
            <div className="rounded-lg border border-border p-3">
              <div className="text-[11px] text-muted-foreground">Business Engagement</div>
              <div className="font-semibold">{fmtPercent(analyzeResult.latest_business_engagement_rate)}</div>
            </div>
            <div className="rounded-lg border border-border p-3">
              <div className="text-[11px] text-muted-foreground">Sector Engagement</div>
              <div className="font-semibold">{fmtPercent(analyzeResult.latest_sector_engagement_rate)}</div>
            </div>
            <div className="rounded-lg border border-border p-3">
              <div className="text-[11px] text-muted-foreground">Difference From Sector</div>
              <div className="font-semibold">{fmtPercent(analyzeResult.difference_from_sector)}</div>
            </div>
          </div>
          <section className="rounded-lg border border-border bg-accent/30 p-3 mb-3">
            <div className="text-xs text-muted-foreground mb-1">Message</div>
            <p className="text-sm">{analyzeResult.message}</p>
          </section>
          <section className="rounded-lg border border-border p-3">
            <div className="text-xs text-muted-foreground mb-2">CSV Outputs</div>
            {analyzeResult.csv_outputs.length === 0 ? (
              <p className="text-sm text-muted-foreground">No files returned.</p>
            ) : (
              <ul className="space-y-1 text-xs">
                {analyzeResult.csv_outputs.map((path) => (
                  <li key={path} className="font-mono break-all">{path}</li>
                ))}
              </ul>
            )}
          </section>
        </motion.div>
      )}

      {statusResult && (
        <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} className="rounded-2xl border border-border bg-card shadow-card p-5">
          <h3 className="text-sm font-semibold mb-3">Status Single Insights</h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-2 mb-4">
            <div className="rounded-lg border border-border p-3"><div className="text-[11px] text-muted-foreground">Business</div><div className="font-semibold truncate">{statusResult.business_name || "-"}</div></div>
            <div className="rounded-lg border border-border p-3"><div className="text-[11px] text-muted-foreground">Trend</div><div><span className={`inline-flex rounded-full px-2 py-0.5 text-xs border ${trendTone(statusResult.trend)}`}>{statusResult.trend}</span></div></div>
            <div className="rounded-lg border border-border p-3"><div className="text-[11px] text-muted-foreground">Comparison</div><div className="font-semibold truncate">{statusResult.comparison_label}</div></div>
            <div className="rounded-lg border border-border p-3"><div className="text-[11px] text-muted-foreground">Difference</div><div className="font-semibold">{fmtPercent(statusResult.difference_from_sector)}</div></div>
          </div>
          <section className="rounded-lg border border-border bg-accent/30 p-3 mb-3">
            <div className="text-xs text-muted-foreground mb-1">Message</div>
            <p className="text-sm">{statusResult.message}</p>
          </section>
          <section className="rounded-lg border border-border p-3">
            <div className="text-xs text-muted-foreground mb-2">CSV Outputs</div>
            {statusResult.csv_outputs.length === 0 ? (
              <p className="text-sm text-muted-foreground">No files returned.</p>
            ) : (
              <ul className="space-y-1 text-xs">
                {statusResult.csv_outputs.map((path) => (
                  <li key={path} className="font-mono break-all">{path}</li>
                ))}
              </ul>
            )}
          </section>
        </motion.div>
      )}

      {sectorSummary && (
        <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} className="rounded-2xl border border-border bg-card shadow-card p-5">
          <h3 className="text-sm font-semibold mb-3">Sector Summary Insights</h3>
          {sectorSummary.length === 0 ? (
            <p className="text-sm text-muted-foreground">No sectors returned.</p>
          ) : (
            <div className="overflow-x-auto rounded-lg border border-border">
              <table className="w-full text-sm">
                <thead className="bg-accent/40 text-xs text-muted-foreground">
                  <tr>
                    <th className="text-left px-3 py-2 font-medium">Sector</th>
                    <th className="text-left px-3 py-2 font-medium">Avg Engagement Rate</th>
                    <th className="text-left px-3 py-2 font-medium">Trend</th>
                  </tr>
                </thead>
                <tbody>
                  {sectorSummary.map((row) => (
                    <tr key={row.sector} className="border-t border-border/70">
                      <td className="px-3 py-2">{row.sector || "-"}</td>
                      <td className="px-3 py-2 font-medium">{fmtPercent(row.avg_engagement_rate)}</td>
                      <td className="px-3 py-2">
                        <span className={`inline-flex rounded-full px-2 py-0.5 text-xs border ${trendTone(row.trend)}`}>
                          {row.trend}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </motion.div>
      )}
    </div>
  );
}
