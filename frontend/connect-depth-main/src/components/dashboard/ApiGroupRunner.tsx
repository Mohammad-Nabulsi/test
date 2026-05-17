import { useMemo, useState } from "react";
import { CheckCircle2, Database, Loader2, Sparkles } from "lucide-react";

type HttpMethod = "GET" | "POST";

type RunContext = {
  datasetPath: string;
  businessName: string;
};

export type EndpointConfig = {
  id: string;
  label: string;
  method: HttpMethod;
  path: string;
  datasetField?: "json_path" | "uploaded_file_path";
  body?: Record<string, unknown>;
  query?: Record<string, string>;
};

type EndpointState = {
  loading: boolean;
  status?: number;
  data?: unknown;
  error?: string;
};

type ApiGroupRunnerProps = {
  title: string;
  subtitle: string;
  endpoints: EndpointConfig[];
  apiBase: string;
  defaultDatasetPath: string;
};

type UploadResponse = {
  dataset_id: string;
};

function formatNumber(value: unknown, digits = 4): string {
  if (typeof value !== "number" || !Number.isFinite(value)) return "-";
  return value.toFixed(digits);
}

function formatRate(value: unknown): string {
  if (typeof value !== "number" || !Number.isFinite(value)) return "-";
  // Values <= 1 are usually true rates; larger values are likely already scaled.
  if (Math.abs(value) <= 1) return `${(value * 100).toFixed(2)}%`;
  return value.toFixed(4);
}

function formatAny(value: unknown): string {
  if (value === null) return "null";
  if (value === undefined) return "-";
  if (typeof value === "number" && Number.isFinite(value)) return String(value);
  if (typeof value === "boolean") return value ? "true" : "false";
  if (typeof value === "string") return value || "-";
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}

function fieldLabel(key: string): string {
  return key
    .replace(/_/g, " ")
    .replace(/\b\w/g, (char) => char.toUpperCase());
}

function KpiGrid({ items }: { items: Array<{ label: string; value: string | number }> }) {
  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-3">
      {items.map((item) => (
        <div
          key={item.label}
          className="rounded-2xl border border-border/70 bg-gradient-to-b from-background to-muted/20 p-4 shadow-card"
        >
          <div className="text-[11px] uppercase tracking-wide text-muted-foreground">{item.label}</div>
          <div className="mt-1 text-base font-semibold num-display break-all">{item.value}</div>
        </div>
      ))}
    </div>
  );
}

function listToTable(rows: Record<string, unknown>[]) {
  if (!rows.length) return null;
  const keys = Object.keys(rows[0]).slice(0, 12);
  return (
    <div className="overflow-auto rounded-2xl border border-border/80 shadow-card">
      <table className="min-w-full text-xs md:text-sm">
        <thead className="bg-muted/60">
          <tr>
            {keys.map((k) => (
              <th key={k} className="text-left px-3 py-2.5 font-semibold whitespace-nowrap">
                {fieldLabel(k)}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.slice(0, 15).map((row, i) => (
            <tr key={i} className="border-t border-border/70 odd:bg-background even:bg-muted/20">
              {keys.map((k) => (
                <td key={k} className="px-3 py-2 whitespace-nowrap">
                  {formatAny(row[k])}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function SemanticResponse({ endpointId, data, apiBase }: { endpointId: string; data: unknown; apiBase: string }) {
  if (!data || typeof data !== "object") {
    return <div className="text-sm text-muted-foreground">No response payload.</div>;
  }

  if (Array.isArray(data)) {
    const rows = data.filter((x): x is Record<string, unknown> => !!x && typeof x === "object" && !Array.isArray(x));
    return (
      <div className="space-y-3">
        <div className="inline-flex rounded-full border border-border/80 bg-muted/40 px-3 py-1 text-xs text-muted-foreground">
          Returned list count: {data.length}
        </div>
        {rows.length > 0 ? listToTable(rows) : <div className="text-sm text-muted-foreground">Empty list.</div>}
      </div>
    );
  }

  const payload = data as Record<string, unknown>;
  const chartUrl = typeof payload.chart_url === "string" ? payload.chart_url : "";
  const resolvedChartUrl =
    chartUrl && (chartUrl.startsWith("http://") || chartUrl.startsWith("https://"))
      ? chartUrl
      : chartUrl
        ? `${apiBase.replace(/\/$/, "")}/${chartUrl.replace(/^\/+/, "")}`
        : "";

  if (endpointId.includes("generate-rules")) {
    return (
      <KpiGrid
        items={[
          { label: "Status", value: String(payload.status ?? "-") },
          { label: "Positive Rules", value: Number(payload.positive_rules_count ?? 0) },
          { label: "Negative Rules", value: Number(payload.negative_rules_count ?? 0) },
        ]}
      />
    );
  }

  if (endpointId.includes("forecast")) {
    const future = Array.isArray(payload.future_forecast) ? (payload.future_forecast as Record<string, unknown>[]) : [];
    const firstForecast = future[0];
    const lastForecast = future[future.length - 1];
    const endpointVariant = endpointId.includes("static") ? "Static" : "Analyze";
    const allFields = Object.entries(payload);
    const scalarFields = allFields.filter(([key, value]) => key !== "future_forecast" && (value === null || typeof value !== "object"));
    const nestedFields = allFields.filter(([key, value]) => key !== "future_forecast" && value !== null && typeof value === "object");
    return (
      <div className="space-y-5">
        <div className="rounded-2xl border border-border/80 bg-gradient-to-r from-accent/35 via-background to-background p-4 shadow-card">
          <div className="flex flex-wrap items-center gap-2">
            <span className="inline-flex items-center rounded-full border border-border/70 bg-background/80 px-3 py-1 text-xs font-medium">
              <Sparkles className="mr-1 h-3.5 w-3.5 text-[var(--brand)]" />
              {endpointVariant} Forecast
            </span>
            {typeof payload.business_name === "string" && payload.business_name ? (
              <span className="inline-flex items-center rounded-full border border-border/70 bg-background/80 px-3 py-1 text-xs">
                <Database className="mr-1 h-3.5 w-3.5 text-[var(--brand-2)]" />
                {payload.business_name}
              </span>
            ) : null}
          </div>
          <p className="mt-3 text-sm text-muted-foreground leading-6">{String(payload.message ?? "Forecast completed.")}</p>
        </div>

        <KpiGrid
          items={[
            { label: "Endpoint", value: endpointVariant },
            { label: "Business", value: String(payload.business_name ?? "-") },
            { label: "Model", value: String(payload.best_model ?? "-") },
            { label: "Horizon (weeks)", value: Number(payload.forecast_horizon_weeks ?? future.length ?? 0) },
            { label: "MAE", value: formatNumber(payload.best_MAE, 4) },
            { label: "RMSE", value: formatNumber(payload.best_RMSE, 4) },
            { label: "First Forecast Date", value: String(firstForecast?.date ?? "-") },
            { label: "Last Forecast Date", value: String(lastForecast?.date ?? "-") },
          ]}
        />

        <div className="rounded-2xl border border-border/80 bg-muted/20 p-4 shadow-card">
          <div className="text-xs font-semibold tracking-wide text-muted-foreground mb-2">All Scalar Response Fields</div>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
            {scalarFields.map(([key, value]) => (
              <div key={key} className="rounded-xl border border-border/70 bg-background p-3">
                <div className="text-[11px] uppercase tracking-wide text-muted-foreground">{fieldLabel(key)}</div>
                <div className="mt-1 text-sm font-medium break-all">{formatAny(value)}</div>
              </div>
            ))}
          </div>
        </div>

        {nestedFields.length > 0 && (
          <div className="rounded-2xl border border-border/80 bg-muted/20 p-4 shadow-card">
            <div className="text-xs font-semibold tracking-wide text-muted-foreground mb-2">Nested Response Fields</div>
            <div className="space-y-3">
              {nestedFields.map(([key, value]) => (
                <div key={key} className="rounded-xl border border-border/70 bg-background p-3">
                  <div className="text-[11px] uppercase tracking-wide text-muted-foreground mb-2">{fieldLabel(key)}</div>
                  <pre className="text-xs leading-6 whitespace-pre-wrap break-words overflow-auto">{formatAny(value)}</pre>
                </div>
              ))}
            </div>
          </div>
        )}

        {resolvedChartUrl ? (
          <div className="rounded-2xl border border-border/80 bg-background p-4 shadow-card">
            <div className="text-xs font-semibold tracking-wide text-muted-foreground mb-3">Forecast Chart</div>
            <img src={resolvedChartUrl} alt="forecast chart" className="rounded-xl border border-border max-h-80 object-contain bg-background w-full" />
          </div>
        ) : (
          <div className="rounded-xl border border-dashed border-border/80 bg-muted/20 p-3 text-xs text-muted-foreground">
            No chart URL returned (`chart_url` is empty).
          </div>
        )}

        {future.length > 0 && (
          <div className="rounded-2xl border border-border/80 bg-background p-4 shadow-card">
            <div className="flex items-center justify-between mb-3">
              <div className="text-sm font-semibold">Future Forecast</div>
              <span className="inline-flex items-center rounded-full border border-border/80 bg-muted/30 px-2.5 py-1 text-xs text-muted-foreground">
                {future.length} rows
              </span>
            </div>
            <div className="overflow-auto rounded-xl border border-border/80">
              <table className="min-w-full text-xs md:text-sm">
                <thead className="bg-muted/60">
                  <tr>
                    <th className="text-left px-3 py-2.5 font-semibold whitespace-nowrap">Date</th>
                    <th className="text-left px-3 py-2.5 font-semibold whitespace-nowrap">Predicted Engagement</th>
                    <th className="text-left px-3 py-2.5 font-semibold whitespace-nowrap">Raw Value</th>
                  </tr>
                </thead>
                <tbody>
                  {future.map((row, idx) => (
                    <tr
                      key={`${String(row.date ?? "unknown")}-${idx}`}
                      className="border-t border-border/70 odd:bg-background even:bg-muted/20"
                    >
                      <td className="px-3 py-2 whitespace-nowrap">{String(row.date ?? "-")}</td>
                      <td className="px-3 py-2 whitespace-nowrap font-medium">{formatRate(row.predicted_engagement_rate)}</td>
                      <td className="px-3 py-2 whitespace-nowrap">{formatNumber(row.predicted_engagement_rate, 8)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        <details className="rounded-2xl border border-border/80 bg-background p-4 shadow-card">
          <summary className="cursor-pointer text-sm font-semibold">Raw Response JSON</summary>
          <pre className="mt-3 overflow-auto rounded-xl border border-border/70 bg-muted/20 p-3 text-xs leading-6 whitespace-pre-wrap break-words">
            {JSON.stringify(payload, null, 2)}
          </pre>
        </details>
      </div>
    );
  }

  if (endpointId.includes("anomalies-analyze")) {
    const positives = Array.isArray(payload.top_positive_anomalies) ? (payload.top_positive_anomalies as Record<string, unknown>[]) : [];
    const negatives = Array.isArray(payload.top_negative_anomalies) ? (payload.top_negative_anomalies as Record<string, unknown>[]) : [];
    return (
      <div className="space-y-3">
        <KpiGrid
          items={[
            { label: "Business", value: String(payload.business_name ?? "-") },
            { label: "Sector", value: String(payload.sector ?? "-") },
            { label: "Positive anomalies", value: positives.length },
            { label: "Negative anomalies", value: negatives.length },
          ]}
        />
        <div className="text-xs text-muted-foreground">{String(payload.message ?? "")}</div>
        {resolvedChartUrl && <img src={resolvedChartUrl} alt="anomaly chart" className="rounded-lg border border-border max-h-80 object-contain bg-background" />}
        {positives.length > 0 && <div><div className="text-xs font-medium mb-1">Top Positive</div>{listToTable(positives)}</div>}
        {negatives.length > 0 && <div><div className="text-xs font-medium mb-1">Top Negative</div>{listToTable(negatives)}</div>}
      </div>
    );
  }

  if (endpointId.includes("business-momentum")) {
    return (
      <div className="space-y-3">
        <KpiGrid
          items={[
            { label: "Business", value: String(payload.business_name ?? "-") },
            { label: "Sector", value: String(payload.sector ?? "-") },
            { label: "Trend", value: String(payload.trend ?? "-") },
            { label: "Comparison", value: String(payload.comparison_label ?? "-") },
            { label: "Business ER", value: Number(payload.latest_business_engagement_rate ?? 0).toFixed(4) },
            { label: "Sector ER", value: Number(payload.latest_sector_engagement_rate ?? 0).toFixed(4) },
            { label: "Difference", value: Number(payload.difference_from_sector ?? 0).toFixed(4) },
          ]}
        />
        {resolvedChartUrl && <img src={resolvedChartUrl} alt="momentum chart" className="rounded-lg border border-border max-h-80 object-contain bg-background" />}
      </div>
    );
  }

  if (endpointId.includes("next-post") || endpointId.includes("similar-business") || endpointId.includes("benchmark-dashboard")) {
    const cards: Array<{ label: string; value: string | number }> = [];
    if (payload.status) cards.push({ label: "Status", value: String(payload.status) });
    if (payload.summary && typeof payload.summary === "object") {
      const s = payload.summary as Record<string, unknown>;
      if (s.total_recommendations !== undefined) cards.push({ label: "Total Recommendations", value: Number(s.total_recommendations) });
      if (s.estimated_total_impact !== undefined) cards.push({ label: "Estimated Impact", value: Number(s.estimated_total_impact) });
    }
    if (payload.business_summary && typeof payload.business_summary === "object") {
      const b = payload.business_summary as Record<string, unknown>;
      if (b.business_name) cards.push({ label: "Business", value: String(b.business_name) });
      if (b.sector) cards.push({ label: "Sector", value: String(b.sector) });
      if (b.sector_rank !== undefined) cards.push({ label: "Rank", value: Number(b.sector_rank) });
    }
    const recs = Array.isArray(payload.engagement_recommendations)
      ? (payload.engagement_recommendations as Record<string, unknown>[])
      : Array.isArray(payload.recommendation_cards)
        ? (payload.recommendation_cards as Record<string, unknown>[])
        : [];
    return (
      <div className="space-y-3">
        {cards.length > 0 && <KpiGrid items={cards} />}
        {recs.length > 0 && <div><div className="text-xs font-medium mb-1">Recommendations</div>{listToTable(recs)}</div>}
        {Array.isArray(payload.sector_ranking) && (payload.sector_ranking as unknown[]).length > 0 && (
          <div>
            <div className="text-xs font-medium mb-1">Sector Ranking</div>
            {listToTable(payload.sector_ranking as Record<string, unknown>[])}
          </div>
        )}
      </div>
    );
  }

  const entries = Object.entries(payload).slice(0, 12);
  return <KpiGrid items={entries.map(([k, v]) => ({ label: k, value: typeof v === "object" ? "[object]" : String(v) }))} />;
}

export function ApiGroupRunner({ title, subtitle, endpoints, apiBase, defaultDatasetPath }: ApiGroupRunnerProps) {
  const [states, setStates] = useState<Record<string, EndpointState>>({});
  const [globalLoading, setGlobalLoading] = useState(false);
  const [datasetPath, setDatasetPath] = useState(defaultDatasetPath);
  const [businessName, setBusinessName] = useState("Vanilla Palestine");
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);

  const orderedEndpoints = useMemo(() => endpoints, [endpoints]);

  const uploadDataset = async (file: File) => {
    setUploadError(null);
    setUploading(true);
    try {
      const form = new FormData();
      form.append("file", file);
      const res = await fetch(`${apiBase.replace(/\/$/, "")}/api/upload`, { method: "POST", body: form });
      if (!res.ok) {
        throw new Error(await res.text());
      }
      const payload = (await res.json()) as UploadResponse;
      const ext = file.name.toLowerCase().endsWith(".json") ? "json" : "csv";
      const newPath = `storage/raw/${payload.dataset_id}/raw.${ext}`;
      setDatasetPath(newPath);
    } catch (e) {
      setUploadError(e instanceof Error ? e.message : "Upload failed");
    } finally {
      setUploading(false);
    }
  };

  const runOne = async (endpoint: EndpointConfig, ctx: RunContext) => {
    setStates((prev) => ({ ...prev, [endpoint.id]: { loading: true } }));
    try {
      const url = new URL(endpoint.path, `${apiBase.replace(/\/$/, "")}/`);
      const query = { ...(endpoint.query ?? {}) };
      Object.entries(query).forEach(([k, v]) => {
        let mapped = v;
        if (v === "$business_name") mapped = ctx.businessName;
        if (v === "$dataset_path") mapped = ctx.datasetPath;
        url.searchParams.set(k, mapped);
      });

      const init: RequestInit = { method: endpoint.method, headers: { Accept: "application/json" } };
      if (endpoint.method === "POST") {
        init.headers = { ...init.headers, "Content-Type": "application/json" };
        const body = { ...(endpoint.body ?? {}) };
        if (endpoint.datasetField) body[endpoint.datasetField] = ctx.datasetPath;
        init.body = JSON.stringify(body);
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
        const p = payload as Record<string, unknown>;
        if (typeof p.business_name === "string" && p.business_name.trim()) {
          setBusinessName(p.business_name);
        }
      }
      setStates((prev) => ({
        ...prev,
        [endpoint.id]: {
          loading: false,
          status: res.status,
          data: payload,
          error: res.ok ? undefined : "Request returned non-2xx status.",
        },
      }));
    } catch (err) {
      setStates((prev) => ({
        ...prev,
        [endpoint.id]: {
          loading: false,
          error: err instanceof Error ? err.message : "Unknown error",
        },
      }));
    }
  };

  const runAll = async () => {
    setGlobalLoading(true);
    const ctx: RunContext = { datasetPath, businessName };
    for (const ep of orderedEndpoints) {
      // eslint-disable-next-line no-await-in-loop
      await runOne(ep, ctx);
    }
    setGlobalLoading(false);
  };

  return (
    <div className="space-y-6">
      <div className="relative overflow-hidden rounded-3xl border border-border/80 bg-card p-5 md:p-6 shadow-card">
        <div className="pointer-events-none absolute inset-0 bg-mesh opacity-45" />
        <div className="relative">
          <div className="flex items-start justify-between gap-4 flex-wrap">
            <div>
              <div className="inline-flex items-center rounded-full border border-border/70 bg-background/70 px-3 py-1 text-xs text-muted-foreground">
                <Sparkles className="mr-1 h-3.5 w-3.5 text-[var(--brand)]" />
                Semantic API Viewer
              </div>
              <h2 className="font-display text-2xl md:text-3xl mt-3">{title}</h2>
              <p className="text-sm text-muted-foreground mt-1 max-w-3xl">{subtitle}</p>
            </div>
            <button
              type="button"
              onClick={runAll}
              disabled={globalLoading}
              className="h-11 px-5 rounded-xl bg-gradient-brand text-white text-sm font-semibold shadow-glow disabled:opacity-60 inline-flex items-center"
            >
              {globalLoading ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <CheckCircle2 className="mr-2 h-4 w-4" />}
              {globalLoading ? "Running..." : "Run All Endpoints"}
            </button>
          </div>

          <div className="mt-5 grid md:grid-cols-2 gap-3">
            <label className="text-xs">
              <div className="mb-1 text-muted-foreground">Active dataset path</div>
              <input
                value={datasetPath}
                onChange={(e) => setDatasetPath(e.target.value)}
                className="w-full rounded-xl border border-border bg-background/80 px-3 py-2.5 text-sm shadow-sm focus:outline-none focus:ring-2 focus:ring-[var(--brand)]/25"
              />
            </label>
            <label className="text-xs">
              <div className="mb-1 text-muted-foreground">Business name (for status endpoints)</div>
              <input
                value={businessName}
                onChange={(e) => setBusinessName(e.target.value)}
                className="w-full rounded-xl border border-border bg-background/80 px-3 py-2.5 text-sm shadow-sm focus:outline-none focus:ring-2 focus:ring-[var(--brand)]/25"
              />
            </label>
          </div>

          <div className="mt-3 flex flex-wrap items-center gap-3">
            <input
              type="file"
              accept=".csv,.json"
              onChange={(e) => {
                const file = e.target.files?.[0];
                if (file) void uploadDataset(file);
              }}
              className="text-xs file:mr-3 file:rounded-lg file:border-0 file:bg-muted file:px-3 file:py-2 file:text-xs file:font-medium"
            />
            <span className="text-xs text-muted-foreground inline-flex items-center">
              {uploading ? <Loader2 className="mr-1 h-3.5 w-3.5 animate-spin" /> : null}
              {uploading ? "Uploading..." : "Upload file to auto-set dataset path"}
            </span>
          </div>
          {uploadError && <div className="mt-2 text-xs text-red-600">{uploadError}</div>}
        </div>
      </div>

      {orderedEndpoints.map((ep) => {
        const st = states[ep.id];
        const statusOk = !!(st?.status && st.status >= 200 && st.status < 300);
        return (
          <div key={ep.id} className="rounded-3xl border border-border/80 bg-card p-5 md:p-6 space-y-4 shadow-card">
            <div className="flex items-center justify-between gap-3 flex-wrap">
              <div>
                <div className="text-base font-semibold">{ep.label}</div>
                <div className="text-xs text-muted-foreground mt-1">
                  <span className="inline-flex items-center rounded-full border border-border/70 bg-muted/30 px-2 py-0.5 font-mono mr-2">{ep.method}</span>
                  <span className="font-mono">{ep.path}</span>
                </div>
              </div>
              <div className="flex items-center gap-2">
                <div
                  className={`inline-flex items-center rounded-full px-3 py-1 text-xs font-medium border ${
                    statusOk
                      ? "border-emerald-300/60 bg-emerald-500/10 text-emerald-700"
                      : st?.status
                        ? "border-red-300/60 bg-red-500/10 text-red-700"
                        : "border-border/70 bg-muted/30 text-muted-foreground"
                  }`}
                >
                  HTTP {st?.status ?? "-"}
                </div>
                <button
                  type="button"
                  onClick={() => runOne(ep, { datasetPath, businessName })}
                  disabled={st?.loading}
                  className="h-9 px-4 rounded-xl border border-border bg-background hover:bg-muted/40 text-sm font-medium disabled:opacity-60 inline-flex items-center"
                >
                  {st?.loading ? <Loader2 className="mr-2 h-3.5 w-3.5 animate-spin" /> : null}
                  {st?.loading ? "Running..." : "Run"}
                </button>
              </div>
            </div>

            {st?.error && <div className="rounded-xl border border-red-300/40 bg-red-500/10 p-3 text-xs text-red-700">{st.error}</div>}

            <SemanticResponse endpointId={ep.id} data={st?.data} apiBase={apiBase} />
          </div>
        );
      })}
    </div>
  );
}
