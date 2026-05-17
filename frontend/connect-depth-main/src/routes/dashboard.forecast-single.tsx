import { createFileRoute } from "@tanstack/react-router";
import { ForecastSingleDashboard } from "@/components/dashboard/forecast-single/ForecastSingleDashboard";

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000";

export const Route = createFileRoute("/dashboard/forecast-single")({
  component: ForecastSinglePage,
});

function ForecastSinglePage() {
  return (
    <ForecastSingleDashboard
      apiBase={API_BASE}
      defaultDatasetPath="data/processed/vanilla_kpi_dataset.json"
    />
  );
}
