import { Loader2, Upload, X } from "lucide-react";
import { useRef, useState } from "react";

type UploadResponse = {
  dataset_id: string;
};

type SingleApiDatasetCardProps = {
  apiBase: string;
  datasetPath: string;
  onDatasetPathChange: (value: string) => void;
  onRun: () => void;
  runLabel: string;
  running: boolean;
  uploadNote?: string;
};

export function SingleApiDatasetCard({
  apiBase,
  datasetPath,
  onDatasetPathChange,
  onRun,
  runLabel,
  running,
  uploadNote = "الملف الافتراضي الحالي هو vanilla_kpi_dataset.json، وتقدر ترفّع ملف ثاني إذا بدك.",
}: SingleApiDatasetCardProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [file, setFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);

  const uploadDataset = async (selectedFile: File) => {
    setUploading(true);
    setUploadError(null);
    try {
      const form = new FormData();
      form.append("file", selectedFile);
      const res = await fetch(`${apiBase.replace(/\/$/, "")}/api/upload`, { method: "POST", body: form });
      if (!res.ok) {
        throw new Error(await res.text());
      }
      const payload = (await res.json()) as UploadResponse;
      const ext = selectedFile.name.toLowerCase().endsWith(".json") ? "json" : "csv";
      onDatasetPathChange(`storage/raw/${payload.dataset_id}/raw.${ext}`);
      setFile(selectedFile);
    } catch (error) {
      setUploadError(error instanceof Error ? error.message : "فشل رفع الملف");
    } finally {
      setUploading(false);
    }
  };

  const handleFile = (selectedFile: File | null) => {
    if (!selectedFile) return;
    const name = selectedFile.name.toLowerCase();
    if (!name.endsWith(".json") && !name.endsWith(".csv")) {
      setUploadError("يرجى رفع ملف JSON أو CSV فقط");
      return;
    }
    void uploadDataset(selectedFile);
  };

  const clearFile = () => {
    setFile(null);
    setUploadError(null);
    if (inputRef.current) inputRef.current.value = "";
  };

  return (
    <section className="rounded-2xl border border-border bg-card shadow-card p-5">
      <div className="flex flex-col gap-4">
        <div>
          <h3 className="text-sm font-semibold">مصدر البيانات</h3>
          <p className="text-xs text-muted-foreground mt-1">{uploadNote}</p>
        </div>

        <label className="block">
          <span className="text-xs text-muted-foreground">json_path</span>
          <input
            value={datasetPath}
            onChange={(e) => onDatasetPathChange(e.target.value)}
            className="mt-1 w-full rounded-xl border border-border bg-background px-3 py-2.5 text-sm"
          />
        </label>

        {!file ? (
          <div
            onDragOver={(e) => {
              e.preventDefault();
              setDragOver(true);
            }}
            onDragLeave={() => setDragOver(false)}
            onDrop={(e) => {
              e.preventDefault();
              setDragOver(false);
              handleFile(e.dataTransfer.files[0] ?? null);
            }}
            onClick={() => inputRef.current?.click()}
            className={`rounded-2xl border-2 border-dashed p-6 text-center cursor-pointer transition-colors ${
              dragOver ? "border-[var(--brand)] bg-accent/30" : "border-border hover:border-[var(--brand)]/50 hover:bg-accent/10"
            }`}
          >
            <Upload className="h-7 w-7 mx-auto mb-3 text-muted-foreground" />
            <p className="text-sm font-medium">ارفع dataset جديد أو استخدم الملف الافتراضي</p>
            <p className="text-xs text-muted-foreground mt-1">JSON أو CSV</p>
            <input ref={inputRef} type="file" accept=".json,.csv" className="hidden" onChange={(e) => handleFile(e.target.files?.[0] ?? null)} />
          </div>
        ) : (
          <div className="flex items-center gap-4 rounded-xl bg-accent/30 p-4">
            <div className="flex-1 min-w-0">
              <p className="text-sm font-medium truncate">{file.name}</p>
              <p className="text-xs text-muted-foreground">{(file.size / 1024).toFixed(1)} ك.ب</p>
            </div>
            <button onClick={clearFile} className="p-2 rounded-lg hover:bg-muted transition-colors" disabled={uploading || running}>
              <X className="h-4 w-4" />
            </button>
          </div>
        )}

        {uploadError && <div className="rounded-xl border border-destructive/30 bg-destructive/5 p-3 text-sm text-destructive">{uploadError}</div>}

        <div className="flex justify-end">
          <button
            onClick={onRun}
            disabled={running || uploading}
            className="inline-flex items-center gap-2 rounded-xl bg-gradient-brand px-4 py-2.5 text-sm font-medium text-white shadow-glow disabled:opacity-60"
          >
            {(running || uploading) && <Loader2 className="h-4 w-4 animate-spin" />}
            {running ? "جاري التشغيل..." : runLabel}
          </button>
        </div>
      </div>
    </section>
  );
}
