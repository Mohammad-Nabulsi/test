import { createFileRoute } from "@tanstack/react-router";
import { ApiGroupRunner } from "@/components/dashboard/ApiGroupRunner";

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8002";

export const Route = createFileRoute("/dashboard/anomalies-single")({
  component: AnomaliesSinglePage,
});

function AnomaliesSinglePage() {
  return (
    <ApiGroupRunner
      title="Anomalies Single"
      subtitle="Runs anomaly analyze and sector summary endpoints with full response rendering."
      apiBase={API_BASE}
      defaultDatasetPath="data/processed/vanilla_kpi_dataset.json"
      endpoints={[
        {
          id: "anomalies-analyze-single",
          label: "Anomalies Analyze Single",
          method: "POST",
          path: "/api/anomalies/analyze-single",
          datasetField: "uploaded_file_path",
        },
        {
          id: "anomalies-sector-summary-single",
          label: "Anomalies Sector Summary Single",
          method: "GET",
          path: "/api/anomalies/sector-summary-single",
          query: { sector: "restaurants" },
        },
      ]}
    />
  );
}
