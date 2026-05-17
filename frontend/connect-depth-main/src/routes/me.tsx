import { createFileRoute } from "@tanstack/react-router";
import { motion } from "framer-motion";
import { useCallback, useEffect, useRef, useState } from "react";
import { Loader2, Upload, X, MapPin, Layers } from "lucide-react";
import { getPendingDatasetFile } from "@/lib/upload-state";
import { KpiCard } from "@/components/dashboard/KpiCard";

export const Route = createFileRoute("/me")({ component: MePage });

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000";

type MeBusiness = {
  name: string;
  nameEn: string;
  handle: string;
  category: string;
  location: string;
  followers: number;
  followersGrowth: number;
  posts: number;
  avatarColor: string;
};

type MeKpi = {
  key: string;
  label: string;
  value: number | string;
  suffix?: string;
  delta: number;
  spark: number[];
  string?: boolean;
};

type MeResult = {
  business: MeBusiness;
  kpis: MeKpi[];
};

function MePage() {
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<MeResult | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const runAnalysis = useCallback(async (uploadFile: File) => {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const form = new FormData();
      form.append("file", uploadFile);
      const res = await fetch(`${API_BASE}/api/me/upload`, { method: "POST", body: form });
      if (!res.ok) {
        const text = await res.text();
        throw new Error(text || `Server error (${res.status})`);
      }
      setResult((await res.json()) as MeResult);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Me analysis failed");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    const pending = getPendingDatasetFile();
    if (pending) {
      setFile(pending);
      runAnalysis(pending);
    }
  }, [runAnalysis]);

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
      runAnalysis(f);
    },
    [runAnalysis],
  );

  const clearFile = () => {
    setFile(null);
    setResult(null);
    setError(null);
    if (inputRef.current) inputRef.current.value = "";
  };

  const b = result?.business;

  return (
    <div className="min-h-screen bg-background p-6">
      <div className="max-w-6xl mx-auto space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold tracking-tight">Business Profile</h1>
            <p className="text-sm text-muted-foreground mt-1">Upload a dataset to see business KPIs and insights</p>
          </div>
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
                  Analyzing…
                </span>
              )}
            </div>
          )}
          {error && (
            <div className="mt-4 rounded-xl border border-destructive/30 bg-destructive/5 p-3 text-sm text-destructive">{error}</div>
          )}
        </div>

        {loading && (
          <div className="flex items-center justify-center py-20">
            <Loader2 className="h-8 w-8 animate-spin text-[var(--brand)]" />
            <span className="ml-3 text-sm text-muted-foreground">Computing business KPIs…</span>
          </div>
        )}

        {b && (
          <>
            <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} className="rounded-2xl border border-border bg-card shadow-card p-6">
              <div className="flex items-start gap-4">
                <div className={`h-16 w-16 rounded-2xl bg-gradient-to-br ${b.avatarColor} flex items-center justify-center text-white text-2xl font-bold shrink-0`}>
                  {b.name?.charAt(0) ?? "B"}
                </div>
                <div className="flex-1 min-w-0">
                  <h2 className="text-xl font-bold">{b.name}</h2>
                  <p className="text-sm text-muted-foreground">{b.handle}</p>
                  <div className="flex flex-wrap gap-4 mt-3">
                    <span className="inline-flex items-center gap-1.5 text-xs text-muted-foreground">
                      <Layers className="h-3.5 w-3.5" /> {b.category}
                    </span>
                    <span className="inline-flex items-center gap-1.5 text-xs text-muted-foreground">
                      <MapPin className="h-3.5 w-3.5" /> {b.location}
                    </span>
                  </div>
                </div>
                <div className="flex gap-4 text-center">
                  <div>
                    <div className="num-display text-2xl font-semibold">{b.followers.toLocaleString()}</div>
                    <div className="text-[11px] text-muted-foreground">Followers</div>
                    {b.followersGrowth !== 0 && (
                      <div className={`text-[11px] font-medium mt-0.5 ${b.followersGrowth >= 0 ? "text-success" : "text-destructive"}`}>
                        {b.followersGrowth >= 0 ? "+" : ""}{b.followersGrowth}%
                      </div>
                    )}
                  </div>
                  <div>
                    <div className="num-display text-2xl font-semibold">{b.posts}</div>
                    <div className="text-[11px] text-muted-foreground">Posts</div>
                  </div>
                </div>
              </div>
            </motion.div>

            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              {result.kpis.map((k, i) => (
                <KpiCard key={k.key} index={i} label={k.label} value={k.value} suffix={k.suffix} delta={k.delta} spark={k.spark} string={k.string} />
              ))}
            </div>
          </>
        )}
      </div>
    </div>
  );
}
