import { createFileRoute } from "@tanstack/react-router";
import { ApiGroupRunner } from "@/components/dashboard/ApiGroupRunner";

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8002";

export const Route = createFileRoute("/dashboard/forecast-single")({
  component: ForecastSinglePage,
});

function ForecastSinglePage() {
  return (
    <ApiGroupRunner
      title="Forecast Single"
      subtitle="Runs forecast analyze/static endpoints and shows complete responses."
      apiBase={API_BASE}
      defaultDatasetPath="data/processed/vanilla_kpi_dataset.json"
      endpoints={[
        {
          id: "forecast-analyze-single",
          label: "Forecast Analyze Single",
          method: "POST",
          path: "/api/forecast/analyze-single",
          datasetField: "uploaded_file_path",
        },
        {
          id: "forecast-static-single",
          label: "Forecast Static Single",
          method: "GET",
          path: "/api/forecast/static-single",
        },
      ]}
    />
  );
}
