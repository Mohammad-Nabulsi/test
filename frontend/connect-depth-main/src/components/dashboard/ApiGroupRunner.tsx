import { useMemo, useState } from "react";

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

function KpiGrid({ items }: { items: Array<{ label: string; value: string | number }> }) {
  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
      {items.map((item) => (
        <div key={item.label} className="rounded-lg border border-border bg-background/40 p-3">
          <div className="text-[11px] text-muted-foreground">{item.label}</div>
          <div className="text-sm font-semibold mt-1 break-all">{item.value}</div>
        </div>
      ))}
    </div>
  );
}

function listToTable(rows: Record<string, unknown>[]) {
  if (!rows.length) return null;
  const keys = Object.keys(rows[0]).slice(0, 8);
  return (
    <div className="overflow-auto border border-border rounded-lg">
      <table className="min-w-full text-xs">
        <thead className="bg-background">
          <tr>
            {keys.map((k) => (
              <th key={k} className="text-left px-3 py-2 font-medium whitespace-nowrap">
                {k}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.slice(0, 15).map((row, i) => (
            <tr key={i} className="border-t border-border">
              {keys.map((k) => (
                <td key={k} className="px-3 py-2 whitespace-nowrap">
                  {String(row[k] ?? "")}
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
    return <div className="text-xs text-muted-foreground">No response payload.</div>;
  }

  if (Array.isArray(data)) {
    const rows = data.filter((x): x is Record<string, unknown> => !!x && typeof x === "object" && !Array.isArray(x));
    return (
      <div className="space-y-2">
        <div className="text-xs text-muted-foreground">Returned list count: {data.length}</div>
        {rows.length > 0 ? listToTable(rows) : <div className="text-xs text-muted-foreground">Empty list.</div>}
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
    return (
      <div className="space-y-3">
        <KpiGrid
          items={[
            { label: "Business", value: String(payload.business_name ?? "-") },
            { label: "Model", value: String(payload.best_model ?? "-") },
            { label: "MAE", value: Number(payload.best_MAE ?? 0).toFixed(4) },
            { label: "RMSE", value: Number(payload.best_RMSE ?? 0).toFixed(4) },
          ]}
        />
        <div className="text-xs text-muted-foreground">{String(payload.message ?? "")}</div>
        {resolvedChartUrl && <img src={resolvedChartUrl} alt="forecast chart" className="rounded-lg border border-border max-h-80 object-contain bg-background" />}
        {future.length > 0 && (
          <div>
            <div className="text-xs font-medium mb-1">Future Forecast</div>
            {listToTable(future)}
          </div>
        )}
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
        const mapped = v === "$business_name" ? ctx.businessName : v;
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
    <div className="space-y-5">
      <div className="rounded-2xl border border-border bg-card p-5">
        <h2 className="font-display text-2xl">{title}</h2>
        <p className="text-sm text-muted-foreground mt-1">{subtitle}</p>

        <div className="mt-4 grid md:grid-cols-2 gap-3">
          <label className="text-xs">
            <div className="mb-1 text-muted-foreground">Active dataset path</div>
            <input
              value={datasetPath}
              onChange={(e) => setDatasetPath(e.target.value)}
              className="w-full rounded-lg border border-border bg-background px-3 py-2 text-sm"
            />
          </label>
          <label className="text-xs">
            <div className="mb-1 text-muted-foreground">Business name (for status endpoints)</div>
            <input
              value={businessName}
              onChange={(e) => setBusinessName(e.target.value)}
              className="w-full rounded-lg border border-border bg-background px-3 py-2 text-sm"
            />
          </label>
        </div>

        <div className="mt-3 flex items-center gap-2">
          <input
            type="file"
            accept=".csv,.json"
            onChange={(e) => {
              const file = e.target.files?.[0];
              if (file) void uploadDataset(file);
            }}
            className="text-xs"
          />
          <span className="text-xs text-muted-foreground">{uploading ? "Uploading..." : "Upload file to auto-set dataset path"}</span>
        </div>
        {uploadError && <div className="mt-2 text-xs text-red-600">{uploadError}</div>}

        <button
          type="button"
          onClick={runAll}
          disabled={globalLoading}
          className="mt-4 h-10 px-4 rounded-lg bg-gradient-brand text-white text-sm font-medium disabled:opacity-60"
        >
          {globalLoading ? "Running..." : "Run All Endpoints"}
        </button>
      </div>

      {orderedEndpoints.map((ep) => {
        const st = states[ep.id];
        return (
          <div key={ep.id} className="rounded-2xl border border-border bg-card p-5 space-y-3">
            <div className="flex items-center justify-between gap-3">
              <div>
                <div className="text-sm font-semibold">{ep.label}</div>
                <div className="text-xs text-muted-foreground">
                  <span className="font-mono">{ep.method}</span> <span className="font-mono">{ep.path}</span>
                </div>
              </div>
              <button
                type="button"
                onClick={() => runOne(ep, { datasetPath, businessName })}
                disabled={st?.loading}
                className="h-9 px-3 rounded-lg border border-border bg-background text-sm disabled:opacity-60"
              >
                {st?.loading ? "Running..." : "Run"}
              </button>
            </div>

            <div className="text-xs">
              <span className="font-medium">HTTP status:</span>{" "}
              <span className={st?.status && st.status >= 200 && st.status < 300 ? "text-green-600" : "text-red-600"}>{st?.status ?? "-"}</span>
            </div>
            {st?.error && <div className="text-xs text-red-600">{st.error}</div>}

            <SemanticResponse endpointId={ep.id} data={st?.data} apiBase={apiBase} />
          </div>
        );
      })}
    </div>
  );
}
