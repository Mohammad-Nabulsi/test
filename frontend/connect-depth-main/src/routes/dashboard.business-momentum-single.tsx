import { createFileRoute } from "@tanstack/react-router";
import { motion } from "framer-motion";
import { useCallback, useEffect, useRef, useState } from "react";
import { AlertCircle, BarChart3, Download, FileText, Loader2, Target, Upload, X } from "lucide-react";
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

function resolveAssetUrl(path?: string | null): string {
  if (!path) return "";
  if (path.startsWith("http://") || path.startsWith("https://")) return path;
  return `${API_BASE.replace(/\/$/, "")}/${path.replace(/^\/+/, "")}`;
}

function fmtPercent(value?: number | null): string {
  if (typeof value !== "number" || !Number.isFinite(value)) return "-";
  return `${(value * 100).toFixed(2)}%`;
}

function trendTone(trend?: string): string {
  const t = (trend ?? "").toLowerCase();
  if (t === "improving") return "border-emerald-300/60 bg-emerald-500/10 text-emerald-700";
  if (t === "stable") return "border-blue-300/60 bg-blue-500/10 text-blue-700";
  if (t === "declining") return "border-rose-300/60 bg-rose-500/10 text-rose-700";
  if (t === "inconsistent") return "border-amber-300/60 bg-amber-500/10 text-amber-700";
  return "border-border bg-muted/30 text-muted-foreground";
}

function comparisonTone(label?: string): string {
  const t = (label ?? "").toLowerCase();
  if (t.includes("above")) return "border-emerald-300/60 bg-emerald-500/10 text-emerald-700";
  if (t.includes("below")) return "border-rose-300/60 bg-rose-500/10 text-rose-700";
  if (t.includes("similar")) return "border-blue-300/60 bg-blue-500/10 text-blue-700";
  return "border-border bg-muted/30 text-muted-foreground";
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

function MetricCard({ label, value, helper }: { label: string; value: string; helper?: string }) {
  return (
    <div className="rounded-2xl border border-border/80 bg-background p-4 shadow-card">
      <div className="text-xs text-muted-foreground">{label}</div>
      <div className="mt-1 text-lg font-semibold num-display truncate">{value}</div>
      {helper ? <div className="mt-1 text-xs text-muted-foreground">{helper}</div> : null}
    </div>
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

function MomentumResultPanel({ title, result }: { title: string; result: MomentumResult }) {
  const chartSrc = resolveAssetUrl(result.chart_url);

  return (
    <motion.section initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} className="rounded-3xl border border-border bg-card p-5 shadow-card space-y-5">
      <div className="flex items-start justify-between gap-3 flex-wrap">
        <div>
          <h3 className="text-xl font-semibold font-display">{title}</h3>
          <p className="mt-1 text-sm text-muted-foreground">{result.business_name || "Unknown business"} momentum summary</p>
        </div>
        <div className="flex flex-wrap gap-2">
          <span className={`rounded-full border px-3 py-1 text-xs font-medium ${trendTone(result.trend)}`}>
            {result.trend || "Unknown trend"}
          </span>
          <span className={`rounded-full border px-3 py-1 text-xs font-medium ${comparisonTone(result.comparison_label)}`}>
            {result.comparison_label || "No comparison"}
          </span>
        </div>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-3">
        <MetricCard label="Business" value={result.business_name || "-"} />
        <MetricCard label="Sector" value={result.sector || "-"} />
        <MetricCard label="Trend" value={result.trend || "-"} />
        <MetricCard label="Comparison" value={result.comparison_label || "-"} />
        <MetricCard label="Business Engagement" value={fmtPercent(result.latest_business_engagement_rate)} helper="Latest engagement rate" />
        <MetricCard label="Sector Engagement" value={fmtPercent(result.latest_sector_engagement_rate)} helper="Latest sector benchmark" />
        <MetricCard label="Difference From Sector" value={fmtPercent(result.difference_from_sector)} helper="Business minus sector" />
        <MetricCard label="Chart Status" value={chartSrc ? "Available" : "Not generated"} />
      </div>

      <div className="rounded-2xl border border-border/80 bg-muted/20 p-4">
        <div className="text-xs text-muted-foreground">Message</div>
        <p className="mt-1 text-sm leading-6">{result.message || "Momentum analysis completed."}</p>
      </div>

      <section className="rounded-2xl border border-border/80 bg-background p-4 shadow-card">
        <h4 className="mb-3 text-sm font-semibold">Momentum Chart</h4>
        {chartSrc ? (
          <img src={chartSrc} alt="Business momentum chart" className="w-full max-h-[460px] rounded-xl border border-border/70 bg-muted/20 object-contain" />
        ) : (
          <EmptyPanel title="No chart generated yet." message="The backend did not return a chart URL for this result." />
        )}
      </section>

      <CsvOutputs outputs={result.csv_outputs} />
    </motion.section>
  );
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
    setError(null);
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
    setError(null);
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

      await runSectorSummary();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Analyze failed");
    } finally {
      setLoadingAnalyze(false);
    }
  }, [runSectorSummary]);

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

  const hasResults = Boolean(analyzeResult || statusResult || sectorSummary);

  return (
    <div className="space-y-5">
      <section className="rounded-3xl border border-border bg-card p-5 shadow-card">
        <div className="mb-4 flex items-start justify-between gap-3 flex-wrap">
          <div>
            <h2 className="text-xl font-semibold font-display">Business Momentum</h2>
            <p className="mt-1 text-sm text-muted-foreground">Compare a business trend against its sector benchmark.</p>
          </div>
          {analyzeResult ? (
            <span className="rounded-full border border-emerald-300/50 bg-emerald-500/10 px-3 py-1 text-xs font-medium text-emerald-700">
              Analysis complete
            </span>
          ) : null}
        </div>

        {!loadingAnalyze && !analyzeResult && !file ? (
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

        {error && (
          <div className="mt-4 rounded-xl border border-destructive/30 bg-destructive/5 p-3 text-sm text-destructive flex items-start gap-2">
            <AlertCircle className="mt-0.5 h-4 w-4" />
            <span>{error}</span>
          </div>
        )}
      </section>

      <section className="rounded-3xl border border-border bg-card p-5 shadow-card">
        <h3 className="text-sm font-semibold mb-3">Additional Views</h3>
        <div className="flex flex-wrap items-center gap-2">
          <input
            value={businessNameInput}
            onChange={(e) => setBusinessNameInput(e.target.value)}
            className="w-full md:w-80 rounded-xl border border-border bg-background px-3 py-2 text-sm"
            placeholder="Business name for status-single"
          />
          <button
            onClick={() => runStatusSingle(businessNameInput)}
            disabled={loadingStatus}
            className="px-4 py-2 rounded-xl border border-border text-sm hover:bg-accent disabled:opacity-60 inline-flex items-center gap-2"
          >
            {loadingStatus ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Target className="h-3.5 w-3.5" />}
            {loadingStatus ? "Loading status..." : "Load status"}
          </button>
          <button
            onClick={runSectorSummary}
            disabled={loadingSector}
            className="px-4 py-2 rounded-xl border border-border text-sm hover:bg-accent disabled:opacity-60 inline-flex items-center gap-2"
          >
            {loadingSector ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <BarChart3 className="h-3.5 w-3.5" />}
            {loadingSector ? "Loading sectors..." : "Load sector summary"}
          </button>
        </div>
      </section>

      {!loadingAnalyze && !hasResults ? (
        <EmptyPanel title="No business momentum results yet" message="Upload a dataset or load a business status to view momentum insights." />
      ) : null}

      {analyzeResult ? <MomentumResultPanel title="Analyze Single Insights" result={analyzeResult} /> : null}

      {statusResult ? <MomentumResultPanel title="Status Single Insights" result={statusResult} /> : null}

      {sectorSummary ? (
        <motion.section initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} className="rounded-3xl border border-border bg-card p-5 shadow-card">
          <h3 className="text-xl font-semibold font-display">Sector Summary</h3>
          {sectorSummary.length === 0 ? (
            <div className="mt-4">
              <EmptyPanel title="No sectors returned" message="The backend did not return sector momentum rows." />
            </div>
          ) : (
            <div className="mt-4 overflow-x-auto rounded-xl border border-border/80">
              <table className="w-full text-sm">
                <thead className="bg-muted/60 text-xs text-muted-foreground">
                  <tr>
                    <th className="text-left px-3 py-2.5 font-semibold">Sector</th>
                    <th className="text-left px-3 py-2.5 font-semibold">Avg Engagement Rate</th>
                    <th className="text-left px-3 py-2.5 font-semibold">Trend</th>
                  </tr>
                </thead>
                <tbody>
                  {sectorSummary.map((row) => (
                    <tr key={row.sector} className="border-t border-border/70 hover:bg-muted/25">
                      <td className="px-3 py-2.5 font-medium">{row.sector || "-"}</td>
                      <td className="px-3 py-2.5 num-display">{fmtPercent(row.avg_engagement_rate)}</td>
                      <td className="px-3 py-2.5">
                        <span className={`inline-flex rounded-full border px-2.5 py-1 text-xs font-medium ${trendTone(row.trend)}`}>
                          {row.trend || "-"}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </motion.section>
      ) : null}
    </div>
  );
}
