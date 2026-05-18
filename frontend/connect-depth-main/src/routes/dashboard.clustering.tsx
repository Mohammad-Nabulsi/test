import { createFileRoute } from "@tanstack/react-router";
import { motion } from "framer-motion";
import { useCallback, useEffect, useRef, useState } from "react";
import { Loader2, Upload, X } from "lucide-react";
import { getPendingDatasetFile } from "@/lib/upload-state";

export const Route = createFileRoute("/dashboard/clustering")({ component: Clustering });

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000";

type ClusteringResult = {
  content_cluster_plot: { format: string; image_base64?: string; hover_columns?: string[]; image_path?: string };
  business_cluster_plot: { format: string; image_base64?: string; hover_columns?: string[]; image_path?: string };
  metadata: { content: Record<string, unknown>; business: Record<string, unknown> };
};

type AiBusiness = {
  business_name: string;
  recommendation: string;
  low_performing: { cluster: number; meaning: string; avg_engagement_rate: number; avg_view_rate: number; low_cluster_post_count: number };
  high_performing: { cluster: number; meaning: string; avg_engagement_rate: number; avg_view_rate: number; high_cluster_post_count: number };
};

type AiResult = { file_path: string; total_rows: number; unique_clusters: number[]; businesses: AiBusiness[] };
type BusinessMetric = { feature: string; business_value: number; cluster_avg: number; comparison: string };
type BusinessComparison = { business_name: string; kmeans_cluster: number; summary?: string; metrics: BusinessMetric[] };
type BusinessResult = { file_path: string; total_businesses: number; businesses: BusinessComparison[] };
type UploadResponse = { dataset_id: string };

function Clustering() {
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [loadingAi, setLoadingAi] = useState(false);
  const [loadingBusiness, setLoadingBusiness] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<ClusteringResult | null>(null);
  const [aiResult, setAiResult] = useState<AiResult | null>(null);
  const [businessResult, setBusinessResult] = useState<BusinessResult | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const [originalPath, setOriginalPath] = useState("data/vanilla_kpi_dataset.json");

  const runVisualization = useCallback(async (uploadFile: File) => {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const form = new FormData();
      form.append("file", uploadFile);

      const uploadRes = await fetch(`${API_BASE}/api/upload`, {
        method: "POST",
        body: form,
      });
      if (!uploadRes.ok) {
        const text = await uploadRes.text();
        throw new Error(text || `فشل الرفع (${uploadRes.status})`);
      }

      const uploadData = (await uploadRes.json()) as UploadResponse;
      if (!uploadData.dataset_id) {
        throw new Error("استجابة الرفع لا تحتوي على dataset_id");
      }

      const res = await fetch(
        `${API_BASE}/api/cluster-visualization/${encodeURIComponent(uploadData.dataset_id)}?original_file_path=${encodeURIComponent(originalPath)}&include_base64=true`,
      );
      if (!res.ok) {
        const text = await res.text();
        throw new Error(text || `فشل تحليل التجميع (${res.status})`);
      }

      const data = (await res.json()) as ClusteringResult;
      setResult(data);
    } catch (e) {
      setError(e instanceof Error ? e.message : "فشل تحليل التجميع");
    } finally {
      setLoading(false);
    }
  }, [originalPath]);

  const runContentStyleAi = useCallback(async () => {
    setLoadingAi(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/api/content-style-clustering/analyze?file_path=${encodeURIComponent(originalPath)}`);
      if (!res.ok) {
        const text = await res.text();
        throw new Error(text || `فشل clustering-ai (${res.status})`);
      }
      setAiResult((await res.json()) as AiResult);
    } catch (e) {
      setError(e instanceof Error ? e.message : "فشل clustering-ai");
    } finally {
      setLoadingAi(false);
    }
  }, [originalPath]);

  const runBusinessClustering = useCallback(async () => {
    setLoadingBusiness(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/api/business-clustering/analyze?file_path=${encodeURIComponent(originalPath)}`);
      if (!res.ok) {
        const text = await res.text();
        throw new Error(text || `فشل business-clustering (${res.status})`);
      }
      setBusinessResult((await res.json()) as BusinessResult);
    } catch (e) {
      setError(e instanceof Error ? e.message : "فشل business-clustering");
    } finally {
      setLoadingBusiness(false);
    }
  }, [originalPath]);

  useEffect(() => {
    const pending = getPendingDatasetFile();
    if (pending) {
      setFile(pending);
      runVisualization(pending);
    }
  }, [runVisualization]);

  const handleFile = useCallback((f: File | null) => {
    if (!f) return;
    const name = f.name.toLowerCase();
    if (!name.endsWith(".json") && !name.endsWith(".csv")) {
      setError("يرجى رفع ملف JSON أو CSV فقط");
      return;
    }
    setFile(f);
    setError(null);
    setResult(null);
    runVisualization(f);
  }, [runVisualization]);

  const clearFile = () => {
    setFile(null);
    setResult(null);
    setError(null);
    if (inputRef.current) inputRef.current.value = "";
  };

  return (
    <div className="space-y-5">
      <div className="rounded-2xl border border-border bg-card shadow-card p-5">
        <h3 className="text-sm font-semibold mb-3">رفع ملف لعرض تحليل التجميع</h3>
        <label className="block mb-3">
          <span className="text-xs text-muted-foreground">original_file_path</span>
          <input
            value={originalPath}
            onChange={(e) => setOriginalPath(e.target.value)}
            className="mt-1 w-full rounded-lg border border-border bg-background px-3 py-2 text-sm"
          />
        </label>

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
            <p className="text-sm font-medium">اسحب وأفلت ملف JSON أو CSV هنا</p>
            <p className="text-xs text-muted-foreground mt-1">أو اضغط للتصفح</p>
            <input ref={inputRef} type="file" accept=".json,.csv" className="hidden" onChange={(e) => handleFile(e.target.files?.[0] ?? null)} />
          </div>
        ) : file ? (
          <div className="flex items-center gap-4 p-4 rounded-xl bg-accent/30">
            <div className="flex-1 min-w-0">
              <p className="text-sm font-medium truncate">{file.name}</p>
              <p className="text-xs text-muted-foreground">{(file.size / 1024).toFixed(1)} ك.ب</p>
            </div>
            <button onClick={clearFile} className="p-2 rounded-lg hover:bg-muted transition-colors" disabled={loading}>
              <X className="h-4 w-4" />
            </button>
            {loading && (
              <span className="px-4 py-2 rounded-xl bg-gradient-brand text-white text-sm font-medium flex items-center gap-2">
                <Loader2 className="h-3.5 w-3.5 animate-spin" />
                جاري التحليل...
              </span>
            )}
          </div>
        ) : null}

        {error && <div className="mt-4 rounded-xl border border-destructive/30 bg-destructive/5 p-3 text-sm text-destructive">{error}</div>}
      </div>

      <div className="rounded-2xl border border-border bg-card shadow-card p-5">
        <h3 className="text-sm font-semibold mb-3">تحليلات إضافية بنفس الصفحة</h3>
        <div className="flex flex-wrap gap-2">
          <button onClick={runContentStyleAi} disabled={loadingAi} className="px-4 py-2 rounded-xl border border-border text-sm hover:bg-accent disabled:opacity-60">
            {loadingAi ? "جاري تشغيل clustering-ai..." : "تشغيل clustering-ai"}
          </button>
          <button onClick={runBusinessClustering} disabled={loadingBusiness} className="px-4 py-2 rounded-xl border border-border text-sm hover:bg-accent disabled:opacity-60">
            {loadingBusiness ? "جاري تشغيل business-clustering..." : "تشغيل business-clustering"}
          </button>
        </div>
      </div>

      {loading && (
        <div className="flex items-center justify-center py-16">
          <Loader2 className="h-8 w-8 animate-spin text-[var(--brand)]" />
          <span className="mr-3 text-sm text-muted-foreground">جاري إنشاء مخططات التجميع...</span>
        </div>
      )}

      {result && (
        <>
          <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} className="rounded-2xl border border-border bg-card shadow-card p-5">
            <h3 className="text-sm font-semibold mb-3">مخطط تجميع المحتوى (Content Clusters)</h3>
            {result.content_cluster_plot.image_base64 ? (
              <img src={`data:image/png;base64,${result.content_cluster_plot.image_base64}`} alt="مخطط تجميع المحتوى" className="w-full rounded-xl border border-border/60 bg-white" />
            ) : (
              <div className="text-sm text-destructive">تعذر عرض صورة مخطط المحتوى.</div>
            )}
          </motion.div>

          <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }} className="rounded-2xl border border-border bg-card shadow-card p-5">
            <h3 className="text-sm font-semibold mb-3">مخطط تجميع الأعمال (Business Clusters)</h3>
            {result.business_cluster_plot.image_base64 ? (
              <img src={`data:image/png;base64,${result.business_cluster_plot.image_base64}`} alt="مخطط تجميع الأعمال" className="w-full rounded-xl border border-border/60 bg-white" />
            ) : (
              <div className="text-sm text-destructive">تعذر عرض صورة مخطط الأعمال.</div>
            )}
          </motion.div>
        </>
      )}

      {aiResult && (
        <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} className="rounded-2xl border border-border bg-card shadow-card p-5">
          <h3 className="text-sm font-semibold mb-3">نتيجة clustering-ai</h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-2 mb-4">
            <div className="rounded-lg border border-border p-3"><div className="text-[11px] text-muted-foreground">Rows</div><div className="font-semibold">{aiResult.total_rows}</div></div>
            <div className="rounded-lg border border-border p-3"><div className="text-[11px] text-muted-foreground">Clusters</div><div className="font-semibold">{aiResult.unique_clusters.length}</div></div>
            <div className="rounded-lg border border-border p-3"><div className="text-[11px] text-muted-foreground">Businesses</div><div className="font-semibold">{aiResult.businesses.length}</div></div>
            <div className="rounded-lg border border-border p-3"><div className="text-[11px] text-muted-foreground">File</div><div className="font-semibold truncate">{aiResult.file_path}</div></div>
          </div>
        </motion.div>
      )}

      {businessResult && (
        <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} className="rounded-2xl border border-border bg-card shadow-card p-5">
          <h3 className="text-sm font-semibold mb-3">نتيجة business-clustering</h3>
          <div className="grid grid-cols-2 md:grid-cols-3 gap-2 mb-4">
            <div className="rounded-lg border border-border p-3"><div className="text-[11px] text-muted-foreground">Businesses</div><div className="font-semibold">{businessResult.total_businesses}</div></div>
            <div className="rounded-lg border border-border p-3"><div className="text-[11px] text-muted-foreground">File</div><div className="font-semibold truncate">{businessResult.file_path}</div></div>
          </div>
        </motion.div>
      )}
    </div>
  );
}
