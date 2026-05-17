import { createFileRoute } from "@tanstack/react-router";
import { motion } from "framer-motion";
import { useCallback, useRef, useState } from "react";
import { Loader2, Upload, X } from "lucide-react";

export const Route = createFileRoute("/cluster-visualization")({ component: ClusterVisualization });

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000";

type ClusteringResult = {
  content_cluster_plot: { format: string; image_base64: string; hover_columns?: string[]; image_path?: string };
  business_cluster_plot: { format: string; image_base64: string; hover_columns?: string[]; image_path?: string };
  metadata: { content: Record<string, unknown>; business: Record<string, unknown> };
};

type UploadResponse = { dataset_id: string };

function ClusterVisualization() {
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<ClusteringResult | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const runVisualization = useCallback(async (uploadFile: File) => {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const form = new FormData();
      form.append("file", uploadFile);
      const uploadRes = await fetch(`${API_BASE}/api/upload`, { method: "POST", body: form });
      if (!uploadRes.ok) {
        const text = await uploadRes.text();
        throw new Error(text || `Upload failed (${uploadRes.status})`);
      }
      const uploadData = (await uploadRes.json()) as UploadResponse;
      if (!uploadData.dataset_id) {
        throw new Error("Upload response is missing dataset_id.");
      }

      const originalPath = "data/vanilla_kpi_dataset.json";
      const res = await fetch(
        `${API_BASE}/api/cluster-visualization/${encodeURIComponent(uploadData.dataset_id)}?original_file_path=${encodeURIComponent(originalPath)}&include_base64=true`
      );
      if (!res.ok) {
        const text = await res.text();
        throw new Error(text || `Cluster visualization failed (${res.status})`);
      }
      const data = (await res.json()) as ClusteringResult;
      setResult(data);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Cluster visualization failed");
    } finally {
      setLoading(false);
    }
  }, []);

  const handleFile = useCallback(
    (f: File | null) => {
      if (!f) return;
      const name = f.name.toLowerCase();
      if (!name.endsWith(".json") && !name.endsWith(".csv")) {
        setError("Please upload a JSON or CSV file only");
        return;
      }
      setFile(f);
      setError(null);
      setResult(null);
      runVisualization(f);
    },
    [runVisualization],
  );

  const clearFile = () => {
    setFile(null);
    setResult(null);
    setError(null);
    if (inputRef.current) inputRef.current.value = "";
  };

  return (
    <div className="min-h-screen bg-background p-6">
      <div className="max-w-6xl mx-auto space-y-6">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">Cluster Visualization</h1>
          <p className="text-sm text-muted-foreground mt-1">
            Upload a JSON or CSV dataset to see content and business cluster projections
          </p>
        </div>

        <div className="rounded-2xl border border-border bg-card shadow-card p-5">
          {!file ? (
            <div
              onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
              onDragLeave={() => setDragOver(false)}
              onDrop={(e) => { e.preventDefault(); setDragOver(false); handleFile(e.dataTransfer.files[0] ?? null); }}
              onClick={() => inputRef.current?.click()}
              className={`border-2 border-dashed rounded-xl p-12 text-center cursor-pointer transition-colors ${
                dragOver ? "border-[var(--brand)] bg-accent/30" : "border-border hover:border-[var(--brand)]/50 hover:bg-accent/10"
              }`}
            >
              <Upload className="h-10 w-10 mx-auto mb-4 text-muted-foreground" />
              <p className="text-base font-medium">Drag & drop a JSON or CSV file here</p>
              <p className="text-sm text-muted-foreground mt-1">or click to browse</p>
              <input ref={inputRef} type="file" accept=".json,.csv" className="hidden" onChange={(e) => handleFile(e.target.files?.[0] ?? null)} />
            </div>
          ) : (
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
                  Processing...
                </span>
              )}
            </div>
          )}

          {error && (
            <div className="mt-4 rounded-xl border border-destructive/30 bg-destructive/5 p-3 text-sm text-destructive">
              {error}
            </div>
          )}
        </div>

        {loading && (
          <div className="flex items-center justify-center py-20">
            <Loader2 className="h-8 w-8 animate-spin text-[var(--brand)]" />
            <span className="ml-3 text-sm text-muted-foreground">Generating cluster plots...</span>
          </div>
        )}

        {result && (
          <>
            <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} className="rounded-2xl border border-border bg-card shadow-card p-5">
              <h2 className="text-base font-semibold mb-4">Content Cluster Plot</h2>
              <img
                src={`data:image/png;base64,${result.content_cluster_plot.image_base64}`}
                alt="Content cluster plot"
                className="w-full rounded-xl border border-border/60 bg-white"
              />
            </motion.div>

            <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }} className="rounded-2xl border border-border bg-card shadow-card p-5">
              <h2 className="text-base font-semibold mb-4">Business Cluster Plot</h2>
              <img
                src={`data:image/png;base64,${result.business_cluster_plot.image_base64}`}
                alt="Business cluster plot"
                className="w-full rounded-xl border border-border/60 bg-white"
              />
            </motion.div>
          </>
        )}
      </div>
    </div>
  );
}


