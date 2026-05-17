import { useMemo, useState } from "react";
import {
  AlertCircle,
  AreaChart,
  BarChart3,
  CalendarDays,
  CheckCircle2,
  Database,
  Gauge,
  Loader2,
  Sparkles,
  TrendingUp,
  Upload,
} from "lucide-react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

type ForecastRowRaw = {
  date: string;
  predicted_engagement_rate: number;
};

type ForecastResponse = {
  business_name: string;
  best_model: string;
  forecast_horizon_weeks: number;
  best_MAE: number;
  best_RMSE: number;
  message: string;
  future_forecast: ForecastRowRaw[];
  chart_url: string;
};

type EndpointState = {
  loading: boolean;
  status?: number;
  error?: string;
  data?: ForecastResponse;
  raw?: unknown;
};

type EndpointConfig = {
  id: "analyze" | "static";
  title: string;
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

const ENDPOINTS: EndpointConfig[] = [
  {
    id: "analyze",
    title: "Analyze Forecast",
    method: "POST",
    path: "/api/forecast/analyze-single",
  },
  {
    id: "static",
    title: "Static Forecast",
    method: "GET",
    path: "/api/forecast/static-single",
  },
];

function formatDate(input: string): string {
  const d = new Date(input);
  if (Number.isNaN(d.getTime())) return input;
  return new Intl.DateTimeFormat("en-GB", { day: "2-digit", month: "short", year: "numeric" }).format(d);
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

function resolveChartUrl(apiBase: string, chartUrl: string | undefined): string {
  if (!chartUrl) return "";
  if (chartUrl.startsWith("http://") || chartUrl.startsWith("https://")) return chartUrl;
  return `${apiBase.replace(/\/$/, "")}/${chartUrl.replace(/^\/+/, "")}`;
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
    insights.push("No backend chart URL was returned, so visuals are generated directly from future_forecast.");
  }
  return insights.slice(0, 5);
}

function EmptyForecastState({ title }: { title: string }) {
  return (
    <div className="rounded-2xl border border-dashed border-border/80 bg-muted/20 p-8 text-center">
      <AreaChart className="mx-auto h-8 w-8 text-muted-foreground" />
      <h4 className="mt-3 text-sm font-semibold">{title}</h4>
      <p className="mt-1 text-sm text-muted-foreground">Run this endpoint to generate forecast analytics, charts, and insights.</p>
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
      <div className="mt-2 text-lg font-semibold num-display">{value}</div>
      <div className="mt-1 text-xs text-muted-foreground">{helper}</div>
    </div>
  );
}

function ForecastTrendChart({ rows }: { rows: NormalizedForecastRow[] }) {
  if (!rows.length) return <EmptyForecastState title="Predicted Engagement Trend" />;
  return (
    <div className="rounded-2xl border border-border/80 bg-background p-4 shadow-card">
      <h4 className="text-sm font-semibold mb-3">Predicted Engagement Trend</h4>
      <div className="h-72">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={rows} margin={{ top: 8, right: 16, left: 8, bottom: 8 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" />
            <XAxis dataKey="compactDate" tick={{ fontSize: 11 }} minTickGap={20} />
            <YAxis tickFormatter={(v) => formatPrediction(Number(v))} tick={{ fontSize: 11 }} width={80} />
            <Tooltip
              formatter={(v: number) => [formatPrediction(v), "Predicted Engagement"]}
              labelFormatter={(label) => `Date: ${label}`}
              contentStyle={{ borderRadius: 12, borderColor: "var(--color-border)" }}
            />
            <Line
              type="monotone"
              dataKey="chartValue"
              stroke="url(#forecastLineGradient)"
              strokeWidth={3}
              dot={{ r: 3, fill: "var(--brand)" }}
              activeDot={{ r: 5 }}
            />
            <defs>
              <linearGradient id="forecastLineGradient" x1="0" y1="0" x2="1" y2="0">
                <stop offset="0%" stopColor="var(--brand)" />
                <stop offset="100%" stopColor="var(--brand-3)" />
              </linearGradient>
            </defs>
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

function ForecastBarDistribution({ rows }: { rows: NormalizedForecastRow[] }) {
  if (!rows.length) return <EmptyForecastState title="Weekly Forecast Distribution" />;
  return (
    <div className="rounded-2xl border border-border/80 bg-background p-4 shadow-card">
      <h4 className="text-sm font-semibold mb-3">Weekly Forecast Distribution</h4>
      <div className="h-72">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={rows} margin={{ top: 8, right: 16, left: 8, bottom: 8 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" />
            <XAxis dataKey="compactDate" tick={{ fontSize: 11 }} minTickGap={20} />
            <YAxis tickFormatter={(v) => formatPrediction(Number(v))} tick={{ fontSize: 11 }} width={80} />
            <Tooltip
              formatter={(v: number) => [formatPrediction(v), "Predicted Engagement"]}
              labelFormatter={(label) => `Date: ${label}`}
              contentStyle={{ borderRadius: 12, borderColor: "var(--color-border)" }}
            />
            <Bar dataKey="chartValue" fill="url(#forecastBarGradient)" radius={[6, 6, 0, 0]} />
            <defs>
              <linearGradient id="forecastBarGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="var(--brand-2)" />
                <stop offset="100%" stopColor="var(--brand-3)" />
              </linearGradient>
            </defs>
          </BarChart>
        </ResponsiveContainer>
      </div>
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
              <th className="text-left px-3 py-2.5 font-semibold">Raw Value</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={`${row.week}-${row.date}`} className="border-t border-border/70 hover:bg-muted/25 transition-colors">
                <td className="px-3 py-2 num-display">{row.week}</td>
                <td className="px-3 py-2">{row.compactDate}</td>
                <td className="px-3 py-2 font-semibold">{row.displayValue}</td>
                <td className="px-3 py-2">{formatMetric(row.rawValue, 8)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function RawJsonDetails({ state, endpoint }: { state: EndpointState; endpoint: EndpointConfig }) {
  return (
    <details className="rounded-2xl border border-border/80 bg-muted/15 p-4">
      <summary className="cursor-pointer text-sm font-semibold">Developer Details</summary>
      <div className="mt-3 space-y-3 text-xs">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-2">
          <div className="rounded-lg border border-border/70 bg-background p-2.5">
            <div className="text-muted-foreground">HTTP Status</div>
            <div className="font-semibold mt-1">{state.status ?? "-"}</div>
          </div>
          <div className="rounded-lg border border-border/70 bg-background p-2.5">
            <div className="text-muted-foreground">Method</div>
            <div className="font-semibold mt-1">{endpoint.method}</div>
          </div>
          <div className="rounded-lg border border-border/70 bg-background p-2.5">
            <div className="text-muted-foreground">Endpoint</div>
            <div className="font-semibold mt-1 font-mono">{endpoint.path}</div>
          </div>
        </div>
        <pre className="overflow-auto rounded-xl border border-border/70 bg-background p-3 text-xs leading-6 whitespace-pre-wrap break-words">
          {JSON.stringify(state.raw ?? state.data ?? {}, null, 2)}
        </pre>
      </div>
    </details>
  );
}

function ForecastEndpointCard({
  endpoint,
  state,
  onRun,
  apiBase,
}: {
  endpoint: EndpointConfig;
  state: EndpointState;
  onRun: () => Promise<void>;
  apiBase: string;
}) {
  const rows = useMemo(() => normalizeForecastRows(state.data?.future_forecast), [state.data?.future_forecast]);
  const stats = useMemo(() => getPredictionStats(rows), [rows]);
  const quality = useMemo(
    () => getForecastQuality(state.data?.best_MAE, state.data?.best_RMSE),
    [state.data?.best_MAE, state.data?.best_RMSE],
  );
  const insights = useMemo(() => {
    if (!state.data) return [];
    return buildInsights(state.data, rows, stats, quality);
  }, [quality, rows, state.data, stats]);

  const statusOk = !!(state.status && state.status >= 200 && state.status < 300);
  const firstDate = rows[0]?.compactDate ?? "-";
  const lastDate = rows[rows.length - 1]?.compactDate ?? "-";
  const avgPrediction = stats ? formatPrediction(stats.avg) : "-";
  const peakPrediction = stats ? formatPrediction(stats.peak) : "-";
  const lowPrediction = stats ? formatPrediction(stats.low) : "-";
  const backendChartSrc = resolveChartUrl(apiBase, state.data?.chart_url);

  return (
    <section className="rounded-3xl border border-border/80 bg-card p-5 md:p-6 shadow-card space-y-4">
      <header className="flex items-center justify-between gap-3 flex-wrap">
        <div>
          <h3 className="text-xl font-semibold font-display">{endpoint.title}</h3>
          <p className="text-xs text-muted-foreground mt-1">
            <span className="inline-flex items-center rounded-full border border-border/70 bg-muted/30 px-2 py-0.5 font-mono mr-2">
              {endpoint.method}
            </span>
            <span className="font-mono">{endpoint.path}</span>
          </p>
        </div>
        <div className="flex items-center gap-2">
          <span
            className={`inline-flex items-center rounded-full px-3 py-1 text-xs font-medium border ${
              statusOk
                ? "border-emerald-300/60 bg-emerald-500/10 text-emerald-700"
                : state.status
                  ? "border-rose-300/60 bg-rose-500/10 text-rose-700"
                  : "border-border/70 bg-muted/30 text-muted-foreground"
            }`}
          >
            HTTP {state.status ?? "-"}
          </span>
          <button
            type="button"
            onClick={() => void onRun()}
            disabled={state.loading}
            className="h-9 px-4 rounded-xl border border-border bg-background hover:bg-muted/35 transition-colors text-sm font-medium disabled:opacity-60 inline-flex items-center"
          >
            {state.loading ? <Loader2 className="mr-2 h-3.5 w-3.5 animate-spin" /> : null}
            {state.loading ? "Running..." : "Run"}
          </button>
        </div>
      </header>

      {state.error ? (
        <div className="rounded-2xl border border-rose-300/50 bg-rose-500/10 p-4 text-rose-700 text-sm flex items-start gap-2">
          <AlertCircle className="h-4 w-4 mt-0.5" />
          <span>{state.error}</span>
        </div>
      ) : null}

      {state.loading && !state.data ? (
        <div className="rounded-2xl border border-border/80 bg-muted/20 p-8 flex items-center justify-center text-sm text-muted-foreground">
          <Loader2 className="h-5 w-5 animate-spin mr-2" />
          Running forecast endpoint...
        </div>
      ) : null}

      {!state.loading && !state.data && !state.error ? <EmptyForecastState title={endpoint.title} /> : null}

      {state.data ? (
        <div className="space-y-4">
          <div className="rounded-2xl border border-border/80 bg-gradient-to-r from-accent/35 via-background to-background p-4 shadow-card">
            <div className="flex items-center flex-wrap gap-2">
              <span className="inline-flex items-center rounded-full border border-border/70 bg-background/80 px-3 py-1 text-xs font-medium">
                <Database className="h-3.5 w-3.5 mr-1 text-[var(--brand)]" />
                {state.data.business_name || "Unknown business"}
              </span>
              <span className={`inline-flex rounded-full border px-3 py-1 text-xs font-medium ${qualityToneClasses(quality.tone)}`}>
                Forecast Quality: {quality.label}
              </span>
            </div>
            <p className="mt-3 text-sm text-muted-foreground">{state.data.message || "Forecast completed."}</p>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-3">
            <ForecastMetricCard icon={<CalendarDays className="h-4 w-4" />} label="Forecast Horizon" value={`${state.data.forecast_horizon_weeks ?? rows.length} weeks`} helper="Forecast window length" />
            <ForecastMetricCard icon={<TrendingUp className="h-4 w-4" />} label="Avg Prediction" value={avgPrediction} helper="Average expected engagement" />
            <ForecastMetricCard icon={<BarChart3 className="h-4 w-4" />} label="Peak Prediction" value={peakPrediction} helper="Highest expected weekly value" />
            <ForecastMetricCard icon={<BarChart3 className="h-4 w-4" />} label="Lowest Prediction" value={lowPrediction} helper="Lowest expected weekly value" />
            <ForecastMetricCard icon={<Gauge className="h-4 w-4" />} label="MAE" value={formatMetric(state.data.best_MAE, 4)} helper="Mean Absolute Error" />
            <ForecastMetricCard icon={<Gauge className="h-4 w-4" />} label="RMSE" value={formatMetric(state.data.best_RMSE, 4)} helper="Root Mean Squared Error" />
            <ForecastMetricCard icon={<Sparkles className="h-4 w-4" />} label="Model" value={state.data.best_model || "-"} helper="Selected best model" />
            <ForecastMetricCard icon={<CalendarDays className="h-4 w-4" />} label="Date Range" value={`${firstDate} -> ${lastDate}`} helper="First and last forecast date" />
          </div>

          <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
            <ForecastTrendChart rows={rows} />
            <ForecastBarDistribution rows={rows} />
          </div>

          <ForecastInsights items={insights} />
          <ForecastTable rows={rows} />

          {backendChartSrc ? (
            <div className="rounded-2xl border border-border/80 bg-background p-4 shadow-card">
              <h4 className="text-sm font-semibold mb-3">Backend Forecast Chart</h4>
              <img
                src={backendChartSrc}
                alt={`${endpoint.title} backend forecast chart`}
                className="w-full max-h-[460px] object-contain rounded-xl border border-border/70 bg-muted/20"
              />
              <p className="mt-2 text-xs text-muted-foreground break-all">{state.data.chart_url}</p>
            </div>
          ) : null}

          {state.data.chart_url ? (
            <div className="rounded-2xl border border-border/80 bg-background p-4 shadow-card">
              <h4 className="text-sm font-semibold mb-3">Backend Chart URL</h4>
              <a className="text-sm text-blue-600 hover:underline break-all" href={state.data.chart_url} target="_blank" rel="noreferrer">
                {state.data.chart_url}
              </a>
            </div>
          ) : null}

          <RawJsonDetails state={state} endpoint={endpoint} />
        </div>
      ) : null}
    </section>
  );
}

export function ForecastSingleDashboard({ apiBase, defaultDatasetPath }: { apiBase: string; defaultDatasetPath: string }) {
  const [datasetPath, setDatasetPath] = useState(defaultDatasetPath);
  const [businessName, setBusinessName] = useState("Vanilla Palestine");
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [states, setStates] = useState<Record<EndpointConfig["id"], EndpointState>>({
    analyze: { loading: false },
    static: { loading: false },
  });
  const [runAllLoading, setRunAllLoading] = useState(false);

  const runEndpoint = async (endpoint: EndpointConfig) => {
    setStates((prev) => ({
      ...prev,
      [endpoint.id]: {
        ...prev[endpoint.id],
        loading: true,
        error: undefined,
      },
    }));

    try {
      const url = new URL(endpoint.path, `${apiBase.replace(/\/$/, "")}/`);
      const init: RequestInit = {
        method: endpoint.method,
        headers: { Accept: "application/json" },
      };

      if (endpoint.id === "analyze") {
        init.headers = { ...init.headers, "Content-Type": "application/json" };
        init.body = JSON.stringify({ uploaded_file_path: datasetPath });
      }

      if (endpoint.id === "static") {
        url.searchParams.set("uploaded_file_path", datasetPath);
      }

      const res = await fetch(url.toString(), init);
      const text = await res.text();
      let payload: unknown = text;
      try {
        payload = text ? JSON.parse(text) : {};
      } catch {
        payload = text;
      }

      if (payload && typeof payload === "object" && !Array.isArray(payload)) {
        const p = payload as Partial<ForecastResponse>;
        if (typeof p.business_name === "string" && p.business_name.trim()) {
          setBusinessName(p.business_name);
        }
      }

      setStates((prev) => ({
        ...prev,
        [endpoint.id]: {
          loading: false,
          status: res.status,
          raw: payload,
          data:
            payload && typeof payload === "object" && !Array.isArray(payload)
              ? (payload as ForecastResponse)
              : undefined,
          error: res.ok ? undefined : "Request returned non-2xx status.",
        },
      }));
    } catch (err) {
      setStates((prev) => ({
        ...prev,
        [endpoint.id]: {
          ...prev[endpoint.id],
          loading: false,
          error: err instanceof Error ? err.message : "Unknown error",
        },
      }));
    }
  };

  const runAll = async () => {
    setRunAllLoading(true);
    for (const endpoint of ENDPOINTS) {
      // eslint-disable-next-line no-await-in-loop
      await runEndpoint(endpoint);
    }
    setRunAllLoading(false);
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
                Predict future engagement trends from your selected dataset. Run analyze and static forecast endpoints with customer-ready charts and insights.
              </p>
            </div>
            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={() => void runEndpoint(ENDPOINTS[0])}
                disabled={states.analyze.loading}
                className="h-11 px-5 rounded-xl bg-gradient-brand text-white text-sm font-semibold shadow-glow disabled:opacity-60 inline-flex items-center"
              >
                {states.analyze.loading ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <TrendingUp className="mr-2 h-4 w-4" />}
                Run Forecast
              </button>
              <button
                type="button"
                onClick={() => void runAll()}
                disabled={runAllLoading}
                className="h-11 px-5 rounded-xl border border-border bg-background text-sm font-semibold hover:bg-muted/35 disabled:opacity-60 inline-flex items-center"
              >
                {runAllLoading ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <CheckCircle2 className="mr-2 h-4 w-4" />}
                Run All
              </button>
            </div>
          </div>

          <div className="mt-5 grid grid-cols-1 md:grid-cols-2 gap-3">
            <label className="text-xs">
              <div className="mb-1 text-muted-foreground">Active dataset path</div>
              <input
                value={datasetPath}
                onChange={(e) => setDatasetPath(e.target.value)}
                className="w-full rounded-xl border border-border bg-background/85 px-3 py-2.5 text-sm shadow-sm focus:outline-none focus:ring-2 focus:ring-[var(--brand)]/25"
              />
            </label>
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
            Endpoints:
            <span className="font-mono ml-1 mr-2">{ENDPOINTS[0].method} {ENDPOINTS[0].path}</span>
            <span className="font-mono">{ENDPOINTS[1].method} {ENDPOINTS[1].path}</span>
          </div>
        </div>
      </section>

      <div className="grid grid-cols-1 gap-6">
        <ForecastEndpointCard endpoint={ENDPOINTS[0]} state={states.analyze} onRun={() => runEndpoint(ENDPOINTS[0])} apiBase={apiBase} />
        <ForecastEndpointCard endpoint={ENDPOINTS[1]} state={states.static} onRun={() => runEndpoint(ENDPOINTS[1])} apiBase={apiBase} />
      </div>
    </div>
  );
}
