import { useEffect, useMemo, useState } from "react";
import {
  AlertCircle,
  AreaChart,
  CalendarDays,
  Database,
  Gauge,
  Loader2,
  Sparkles,
  TrendingUp,
  Upload,
} from "lucide-react";

type ForecastRowRaw = {
  date: string;
  predicted_engagement_rate: number;
  lower_bound?: number;
  upper_bound?: number;
};

type TuningSummaryRow = {
  model: string;
  base_model?: string;
  window?: number | null;
  MAE?: number;
  RMSE?: number;
  MAPE?: number | null;
};

type TestPredictionRow = {
  date: string;
  actual_engagement_rate: number;
  predicted_engagement_rate: number;
  residual?: number;
};

type ForecastResponse = {
  business_name: string;
  best_model: string;
  forecast_horizon_weeks: number;
  best_MAE: number;
  best_RMSE: number;
  best_MAPE?: number | null;
  message: string;
  future_forecast: ForecastRowRaw[];
  test_predictions?: TestPredictionRow[];
  tuning_summary?: TuningSummaryRow[];
  forecast_confidence_interval?: Array<{ date: string; lower_bound: number; upper_bound: number }>;
  chart_url: string;
};

type ForecastSectionConfig = {
  title: string;
  description: string;
  method: "GET" | "POST";
  path: string;
};

type UploadResponse = {
  dataset_id: string;
};

type NormalizedForecastRow = {
  week: number;
  date: string;
  compactDate: string;
  rawValue: number;
  displayValue: string;
  displayLower: string;
  displayUpper: string;
  chartValue: number;
};

type PredictionStats = {
  avg: number;
  peak: number;
  low: number;
  delta: number;
  range: number;
  cv: number;
};

type ForecastQuality = {
  label: "Excellent" | "Good" | "Moderate" | "Low";
  tone: "emerald" | "blue" | "amber" | "rose";
  description: string;
};

const UPLOADED_ENDPOINT: ForecastSectionConfig = {
  title: "Forecast Results",
  description: "Runs POST /api/forecast/analyze-single on the selected dataset path.",
  method: "POST",
  path: "/api/forecast/analyze-single",
};

function formatDate(input: string): string {
  const d = new Date(input);
  if (Number.isNaN(d.getTime())) return input;
  return new Intl.DateTimeFormat("en-US", { day: "2-digit", month: "short", year: "numeric" }).format(d);
}

function formatMetric(value: number | undefined | null, digits = 4): string {
  if (typeof value !== "number" || !Number.isFinite(value)) return "-";
  return value.toFixed(digits);
}

function formatPrediction(value: number | undefined | null): string {
  if (typeof value !== "number" || !Number.isFinite(value)) return "-";
  if (value >= 0 && value <= 1) return `${(value * 100).toFixed(2)}%`;
  const abs = Math.abs(value);
  const minFrac = abs >= 10 ? 2 : 3;
  return value.toLocaleString(undefined, { minimumFractionDigits: minFrac, maximumFractionDigits: 4 });
}

function normalizeForecastRows(forecast: ForecastRowRaw[] | undefined): NormalizedForecastRow[] {
  if (!Array.isArray(forecast)) return [];
  return forecast
    .filter((row) => row && typeof row.date === "string" && typeof row.predicted_engagement_rate === "number")
    .map((row, idx) => ({
      week: idx + 1,
      date: row.date,
      compactDate: formatDate(row.date),
      rawValue: row.predicted_engagement_rate,
      chartValue: row.predicted_engagement_rate,
      displayValue: formatPrediction(row.predicted_engagement_rate),
      displayLower: formatPrediction(row.lower_bound),
      displayUpper: formatPrediction(row.upper_bound),
    }));
}

function getPredictionStats(rows: NormalizedForecastRow[]): PredictionStats | null {
  if (!rows.length) return null;
  const values = rows.map((row) => row.rawValue);
  const avg = values.reduce((sum, v) => sum + v, 0) / values.length;
  const peak = Math.max(...values);
  const low = Math.min(...values);
  const delta = values[values.length - 1] - values[0];
  const range = peak - low;
  const variance = values.reduce((sum, v) => sum + (v - avg) ** 2, 0) / values.length;
  const std = Math.sqrt(variance);
  const cv = Math.abs(avg) > 0 ? std / Math.abs(avg) : 0;
  return { avg, peak, low, delta, range, cv };
}

function getForecastQuality(mae: number | undefined, rmse: number | undefined): ForecastQuality {
  const mMae = typeof mae === "number" ? Math.abs(mae) : Number.POSITIVE_INFINITY;
  const mRmse = typeof rmse === "number" ? Math.abs(rmse) : Number.POSITIVE_INFINITY;
  const score = (mMae + mRmse) / 2;
  if (score <= 0.02) {
    return { label: "Excellent", tone: "emerald", description: "Very low error; forecast reliability is strong." };
  }
  if (score <= 0.1) {
    return { label: "Good", tone: "blue", description: "Low error; forecast quality is solid." };
  }
  if (score <= 1) {
    return { label: "Moderate", tone: "amber", description: "Error is moderate; use with practical caution." };
  }
  return { label: "Low", tone: "rose", description: "Higher error; treat this as directional guidance." };
}

function qualityToneClasses(tone: ForecastQuality["tone"]): string {
  if (tone === "emerald") return "border-emerald-300/60 bg-emerald-500/10 text-emerald-700";
  if (tone === "blue") return "border-blue-300/60 bg-blue-500/10 text-blue-700";
  if (tone === "amber") return "border-amber-300/60 bg-amber-500/10 text-amber-700";
  return "border-rose-300/60 bg-rose-500/10 text-rose-700";
}

function resolveChartUrl(apiBase: string, chartUrl: string | null | undefined): string {
  if (!chartUrl) return "";
  if (chartUrl.startsWith("http://") || chartUrl.startsWith("https://")) return chartUrl;
  try {
    const base = new URL(apiBase);
    if (chartUrl.startsWith("/")) {
      return `${base.origin}${chartUrl}`;
    }
    return new URL(chartUrl, `${base.origin}/`).toString();
  } catch {
    return `${apiBase.replace(/\/$/, "")}/${chartUrl.replace(/^\/+/, "")}`;
  }
}

function buildInsights(resp: ForecastResponse, rows: NormalizedForecastRow[], stats: PredictionStats | null, quality: ForecastQuality): string[] {
  const insights: string[] = [];
  insights.push(`Model selected: ${resp.best_model || "unknown model"}.`);
  if (stats) {
    insights.push(`Average expected engagement is ${formatPrediction(stats.avg)} over ${rows.length} forecasted weeks.`);
    if (stats.cv <= 0.03) {
      insights.push(`The projection is highly stable with very low variation across the horizon.`);
    } else if (stats.cv <= 0.1) {
      insights.push(`The projection shows mild variability across the coming weeks.`);
    } else {
      insights.push(`The projection is more volatile; weekly values fluctuate materially.`);
    }
  }
  insights.push(`Forecast quality: ${quality.label}. ${quality.description}`);
  if (!resp.chart_url) {
    insights.push("No chart generated yet.");
  }
  return insights.slice(0, 5);
}

function EmptyForecastState({ title }: { title: string }) {
  return (
    <div className="rounded-2xl border border-dashed border-border/80 bg-muted/20 p-8 text-center">
      <AreaChart className="mx-auto h-8 w-8 text-muted-foreground" />
      <h4 className="mt-3 text-sm font-semibold">{title}</h4>
      <p className="mt-1 text-sm text-muted-foreground">Upload a dataset to run forecasting.</p>
    </div>
  );
}

function ForecastMetricCard({
  icon,
  label,
  value,
  helper,
}: {
  icon: React.ReactNode;
  label: string;
  value: string;
  helper: string;
}) {
  return (
    <div className="rounded-2xl border border-border/70 bg-gradient-to-b from-background to-muted/20 p-4 shadow-card">
      <div className="flex items-center gap-2 text-muted-foreground">
        {icon}
        <span className="text-xs uppercase tracking-wide">{label}</span>
      </div>
      <div className="mt-2 text-lg font-semibold num-display" dir="ltr">{value}</div>
      <div className="mt-1 text-xs text-muted-foreground">{helper}</div>
    </div>
  );
}

function ForecastInsights({ items }: { items: string[] }) {
  return (
    <div className="rounded-2xl border border-border/80 bg-background p-4 shadow-card">
      <h4 className="text-sm font-semibold mb-3">Insights</h4>
      <ul className="space-y-2">
        {items.map((item, idx) => (
          <li key={idx} className="rounded-xl border border-border/70 bg-muted/20 p-3 text-sm leading-6">
            {item}
          </li>
        ))}
      </ul>
    </div>
  );
}

function ForecastTable({ rows }: { rows: NormalizedForecastRow[] }) {
  if (!rows.length) return <EmptyForecastState title="Weekly Forecast Table" />;
  return (
    <div className="rounded-2xl border border-border/80 bg-background p-4 shadow-card">
      <h4 className="text-sm font-semibold mb-3">Weekly Forecast Values</h4>
      <div className="max-h-[380px] overflow-auto rounded-xl border border-border/80">
        <table className="min-w-full text-sm">
          <thead className="sticky top-0 z-10 bg-muted/80 backdrop-blur">
            <tr>
              <th className="text-left px-3 py-2.5 font-semibold">Week</th>
              <th className="text-left px-3 py-2.5 font-semibold">Date</th>
              <th className="text-left px-3 py-2.5 font-semibold">Predicted Engagement</th>
              <th className="text-left px-3 py-2.5 font-semibold">Lower Bound</th>
              <th className="text-left px-3 py-2.5 font-semibold">Upper Bound</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={`${row.week}-${row.date}`} className="border-t border-border/70 hover:bg-muted/25 transition-colors">
                <td className="px-3 py-2 num-display">{row.week}</td>
                <td className="px-3 py-2">{row.compactDate}</td>
                <td className="px-3 py-2 font-semibold">{row.displayValue}</td>
                <td className="px-3 py-2">{row.displayLower}</td>
                <td className="px-3 py-2">{row.displayUpper}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function TuningSummaryTable({ rows }: { rows?: TuningSummaryRow[] }) {
  if (!rows?.length) return null;
  return (
    <div className="rounded-2xl border border-border/80 bg-background p-4 shadow-card">
      <h4 className="text-sm font-semibold mb-3">Model Tuning Summary</h4>
      <div className="max-h-[320px] overflow-auto rounded-xl border border-border/80">
        <table className="min-w-full text-sm">
          <thead className="sticky top-0 z-10 bg-muted/80 backdrop-blur">
            <tr>
              <th className="text-left px-3 py-2.5 font-semibold">Model</th>
              <th className="text-left px-3 py-2.5 font-semibold">Window</th>
              <th className="text-left px-3 py-2.5 font-semibold">MAE</th>
              <th className="text-left px-3 py-2.5 font-semibold">RMSE</th>
              <th className="text-left px-3 py-2.5 font-semibold">MAPE</th>
            </tr>
          </thead>
          <tbody>
            {rows.slice(0, 12).map((row, idx) => (
              <tr key={`${row.model}-${idx}`} className="border-t border-border/70 hover:bg-muted/25 transition-colors">
                <td className="px-3 py-2 font-medium">{row.model}</td>
                <td className="px-3 py-2">{row.window ?? "-"}</td>
                <td className="px-3 py-2">{formatMetric(row.MAE, 4)}</td>
                <td className="px-3 py-2">{formatMetric(row.RMSE, 4)}</td>
                <td className="px-3 py-2">{row.MAPE == null ? "-" : `${row.MAPE.toFixed(2)}%`}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function ForecastResultSection({
  config,
  forecast,
  loading,
  error,
  status,
  apiBase,
  emptyMessage,
}: {
  config: ForecastSectionConfig;
  forecast: ForecastResponse | null;
  loading: boolean;
  error: string | null;
  status: number | null;
  apiBase: string;
  emptyMessage: string;
}) {
  const rows = useMemo(() => normalizeForecastRows(forecast?.future_forecast), [forecast?.future_forecast]);
  const stats = useMemo(() => getPredictionStats(rows), [rows]);
  const quality = useMemo(
    () => getForecastQuality(forecast?.best_MAE, forecast?.best_RMSE),
    [forecast?.best_MAE, forecast?.best_RMSE],
  );
  const insights = useMemo(() => {
    if (!forecast) return [];
    return buildInsights(forecast, rows, stats, quality);
  }, [forecast, quality, rows, stats]);

  const statusOk = !!(status && status >= 200 && status < 300);
  const firstDate = rows[0]?.compactDate ?? "-";
  const lastDate = rows[rows.length - 1]?.compactDate ?? "-";
  const avgPrediction = stats ? formatPrediction(stats.avg) : "-";
  const peakPrediction = stats ? formatPrediction(stats.peak) : "-";
  const lowPrediction = stats ? formatPrediction(stats.low) : "-";
  const backendChartSrc = resolveChartUrl(apiBase, forecast?.chart_url);
  const [chartLoadFailed, setChartLoadFailed] = useState(false);

  useEffect(() => {
    setChartLoadFailed(false);
  }, [backendChartSrc]);

  return (
    <section className="rounded-3xl border border-border/80 bg-card p-5 md:p-6 shadow-card space-y-4">
      <header className="flex items-center justify-between gap-3 flex-wrap">
        <div>
          <h3 className="text-xl font-semibold font-display">{config.title}</h3>
          <p className="mt-1 text-sm text-muted-foreground">{config.description}</p>
          <p className="text-xs text-muted-foreground mt-1">
            <span className="inline-flex items-center rounded-full border border-border/70 bg-muted/30 px-2 py-0.5 font-mono mr-2">
              {config.method}
            </span>
            <span className="font-mono">{config.path}</span>
          </p>
        </div>
        <div className="flex items-center gap-2">
          <span
            className={`inline-flex items-center rounded-full px-3 py-1 text-xs font-medium border ${
              statusOk
                ? "border-emerald-300/60 bg-emerald-500/10 text-emerald-700"
                : status
                  ? "border-rose-300/60 bg-rose-500/10 text-rose-700"
                  : "border-border/70 bg-muted/30 text-muted-foreground"
            }`}
          >
            HTTP {status ?? "-"}
          </span>
        </div>
      </header>

      {error ? (
        <div className="rounded-2xl border border-rose-300/50 bg-rose-500/10 p-4 text-rose-700 text-sm flex items-start gap-2">
          <AlertCircle className="h-4 w-4 mt-0.5" />
          <span>{error}</span>
        </div>
      ) : null}

      {loading && !forecast ? (
        <div className="rounded-2xl border border-border/80 bg-muted/20 p-8 flex items-center justify-center text-sm text-muted-foreground">
          <Loader2 className="h-5 w-5 animate-spin mr-2" />
          Running forecast endpoint...
        </div>
      ) : null}

      {!loading && !forecast && !error ? <EmptyForecastState title={emptyMessage} /> : null}

      {forecast ? (
        <div className="space-y-4">
          <div className="rounded-2xl border border-border/80 bg-gradient-to-r from-accent/35 via-background to-background p-4 shadow-card">
            <div className="flex items-center flex-wrap gap-2">
              <span className="inline-flex items-center rounded-full border border-border/70 bg-background/80 px-3 py-1 text-xs font-medium">
                <Database className="h-3.5 w-3.5 mr-1 text-[var(--brand)]" />
                {forecast.business_name || "Unknown business"}
              </span>
              <span className={`inline-flex rounded-full border px-3 py-1 text-xs font-medium ${qualityToneClasses(quality.tone)}`}>
                Forecast Quality: {quality.label}
              </span>
            </div>
            <p className="mt-3 text-sm text-muted-foreground">{forecast.message || "Forecast completed."}</p>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-3">
            <ForecastMetricCard icon={<CalendarDays className="h-4 w-4" />} label="Forecast Horizon" value={`${forecast.forecast_horizon_weeks ?? rows.length} weeks`} helper="Forecast window length" />
            <ForecastMetricCard icon={<TrendingUp className="h-4 w-4" />} label="Avg Prediction" value={avgPrediction} helper="Average expected engagement" />
            <ForecastMetricCard icon={<TrendingUp className="h-4 w-4" />} label="Peak Prediction" value={peakPrediction} helper="Highest expected weekly value" />
            <ForecastMetricCard icon={<TrendingUp className="h-4 w-4" />} label="Lowest Prediction" value={lowPrediction} helper="Lowest expected weekly value" />
            <ForecastMetricCard icon={<Gauge className="h-4 w-4" />} label="MAE" value={formatMetric(forecast.best_MAE, 4)} helper="Mean Absolute Error" />
            <ForecastMetricCard icon={<Gauge className="h-4 w-4" />} label="RMSE" value={formatMetric(forecast.best_RMSE, 4)} helper="Root Mean Squared Error" />
            <ForecastMetricCard icon={<Gauge className="h-4 w-4" />} label="MAPE" value={forecast.best_MAPE == null ? "-" : `${forecast.best_MAPE.toFixed(2)}%`} helper="Mean Absolute Percentage Error" />
            <ForecastMetricCard icon={<Sparkles className="h-4 w-4" />} label="Model" value={forecast.best_model || "-"} helper="Selected best model" />
            <ForecastMetricCard icon={<CalendarDays className="h-4 w-4" />} label="Date Range" value={`${firstDate} → ${lastDate}`} helper="First and last forecast date" />
          </div>

          <ForecastTable rows={rows} />
          <TuningSummaryTable rows={forecast.tuning_summary} />

          {backendChartSrc && !chartLoadFailed ? (
            <div className="rounded-2xl border border-border/80 bg-background p-4 shadow-card">
              <h4 className="text-sm font-semibold mb-3">Backend Forecast Chart</h4>
              <img
                src={backendChartSrc}
                alt={`${config.title} backend forecast chart`}
                onError={() => setChartLoadFailed(true)}
                className="w-full max-h-[460px] object-contain rounded-xl border border-border/70 bg-muted/20"
              />
            </div>
          ) : (
            <div className="rounded-2xl border border-border/80 bg-background p-4 shadow-card">
              <h4 className="text-sm font-semibold mb-3">Backend Forecast Chart</h4>
              <div className="rounded-2xl border border-dashed border-border/80 bg-muted/20 p-8 text-center">
                <AreaChart className="mx-auto h-8 w-8 text-muted-foreground" />
                <p className="mt-3 text-sm font-medium">
                  {backendChartSrc ? "Chart image could not be loaded." : "No chart generated yet."}
                </p>
              </div>
            </div>
          )}

          <ForecastInsights items={insights} />
        </div>
      ) : null}
    </section>
  );
}

export function ForecastSingleDashboard({ apiBase }: { apiBase: string }) {
  const [datasetPath, setDatasetPath] = useState("");
  const [businessName, setBusinessName] = useState("Vanilla Palestine");
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [uploadedForecast, setUploadedForecast] = useState<ForecastResponse | null>(null);
  const [uploadedLoading, setUploadedLoading] = useState(false);
  const [uploadedError, setUploadedError] = useState<string | null>(null);
  const [uploadedStatus, setUploadedStatus] = useState<number | null>(null);

  const parseForecastResponse = async (res: Response): Promise<ForecastResponse> => {
    const text = await res.text();
    let payload: unknown = {};
    try {
      payload = text ? JSON.parse(text) : {};
    } catch {
      throw new Error(text || "Response was not valid JSON.");
    }

    if (!res.ok) {
      const detail = payload && typeof payload === "object" && "detail" in payload ? String((payload as { detail: unknown }).detail) : "";
      throw new Error(detail || `Request failed (${res.status}).`);
    }

    if (!payload || typeof payload !== "object" || Array.isArray(payload)) {
      throw new Error("Forecast response was not an object.");
    }

    return payload as ForecastResponse;
  };

  const rememberBusinessName = (forecast: ForecastResponse) => {
    if (forecast.business_name?.trim()) {
      setBusinessName(forecast.business_name);
    }
  };

  const runUploadedForecast = async () => {
    if (!datasetPath) {
      setUploadedError("Upload a dataset to run forecasting.");
      return;
    }
    setUploadedLoading(true);
    setUploadedError(null);
    try {
      const url = new URL(UPLOADED_ENDPOINT.path, `${apiBase.replace(/\/$/, "")}/`);
      const res = await fetch(url.toString(), {
        method: UPLOADED_ENDPOINT.method,
        headers: { Accept: "application/json", "Content-Type": "application/json" },
        body: JSON.stringify({ uploaded_file_path: datasetPath }),
      });
      setUploadedStatus(res.status);
      const forecast = await parseForecastResponse(res);
      setUploadedForecast(forecast);
      rememberBusinessName(forecast);
    } catch (err) {
      setUploadedError(err instanceof Error ? err.message : "Uploaded forecast failed.");
    } finally {
      setUploadedLoading(false);
    }
  };

  const uploadDataset = async (file: File) => {
    setUploadError(null);
    setUploading(true);
    try {
      const form = new FormData();
      form.append("file", file);
      const res = await fetch(`${apiBase.replace(/\/$/, "")}/api/upload`, { method: "POST", body: form });
      if (!res.ok) throw new Error(await res.text());
      const payload = (await res.json()) as UploadResponse;
      const ext = file.name.toLowerCase().endsWith(".json") ? "json" : "csv";
      setDatasetPath(`storage/raw/${payload.dataset_id}/raw.${ext}`);
      setUploadedForecast(null);
      setUploadedStatus(null);
      setUploadedError(null);
    } catch (err) {
      setUploadError(err instanceof Error ? err.message : "Upload failed");
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="space-y-6">
      <section className="relative overflow-hidden rounded-3xl border border-border/80 bg-card p-5 md:p-6 shadow-card">
        <div className="pointer-events-none absolute inset-0 bg-mesh opacity-45" />
        <div className="relative">
          <div className="flex items-start justify-between gap-4 flex-wrap">
            <div>
              <div className="inline-flex items-center rounded-full border border-border/70 bg-background/80 px-3 py-1 text-xs text-muted-foreground">
                <Sparkles className="mr-1 h-3.5 w-3.5 text-[var(--brand)]" />
                Forecast API Viewer
              </div>
              <h1 className="mt-3 text-2xl md:text-3xl font-display font-semibold">Forecast Intelligence</h1>
              <p className="mt-2 text-sm text-muted-foreground max-w-3xl">
                Upload a CSV or JSON dataset, then run a forecast with customer-ready metrics, tables, and chart output.
              </p>
            </div>
            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={() => void runUploadedForecast()}
                disabled={uploadedLoading || !datasetPath}
                className="h-11 px-5 rounded-xl bg-gradient-brand text-white text-sm font-semibold shadow-glow disabled:opacity-60 inline-flex items-center"
              >
                {uploadedLoading ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <TrendingUp className="mr-2 h-4 w-4" />}
                Run Forecast
              </button>
            </div>
          </div>

          <div className="mt-5 grid grid-cols-1 md:grid-cols-2 gap-3">
            <div className="rounded-xl border border-border bg-background/85 px-3 py-2.5 text-xs shadow-sm">
              <div className="mb-1 text-muted-foreground">Uploaded dataset path</div>
              <div className="font-mono text-sm break-all">{datasetPath || "Upload a dataset to run forecasting."}</div>
            </div>
            <label className="text-xs">
              <div className="mb-1 text-muted-foreground">Business name</div>
              <input
                value={businessName}
                onChange={(e) => setBusinessName(e.target.value)}
                className="w-full rounded-xl border border-border bg-background/85 px-3 py-2.5 text-sm shadow-sm focus:outline-none focus:ring-2 focus:ring-[var(--brand)]/25"
              />
            </label>
          </div>

          <div className="mt-3 flex items-center gap-3 flex-wrap">
            <label className="inline-flex items-center gap-2 rounded-xl border border-border bg-background px-3 py-2 text-xs hover:bg-muted/25 transition-colors cursor-pointer">
              <Upload className="h-3.5 w-3.5" />
              Upload CSV/JSON
              <input
                type="file"
                accept=".csv,.json"
                className="hidden"
                onChange={(e) => {
                  const file = e.target.files?.[0];
                  if (file) void uploadDataset(file);
                }}
              />
            </label>
            <span className="text-xs text-muted-foreground inline-flex items-center">
              {uploading ? <Loader2 className="mr-1 h-3.5 w-3.5 animate-spin" /> : null}
              {uploading ? "Uploading dataset..." : "Upload updates the active dataset path automatically."}
            </span>
          </div>
          {uploadError ? <div className="mt-2 text-xs text-rose-700">{uploadError}</div> : null}

          <div className="mt-4 rounded-xl border border-border/70 bg-muted/20 p-3 text-[11px] text-muted-foreground">
            Endpoint:
            <span className="font-mono ml-1">{UPLOADED_ENDPOINT.method} {UPLOADED_ENDPOINT.path}</span>
          </div>
        </div>
      </section>

      <div className="grid grid-cols-1 gap-6">
        <ForecastResultSection
          config={UPLOADED_ENDPOINT}
          forecast={uploadedForecast}
          loading={uploadedLoading}
          error={uploadedError}
          status={uploadedStatus}
          apiBase={apiBase}
          emptyMessage="Upload a dataset to run forecasting."
        />
      </div>
    </div>
  );
}
