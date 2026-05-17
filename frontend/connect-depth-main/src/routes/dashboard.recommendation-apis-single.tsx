import { createFileRoute } from "@tanstack/react-router";
import { ApiGroupRunner } from "@/components/dashboard/ApiGroupRunner";

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000";

export const Route = createFileRoute("/dashboard/recommendation-apis-single")({
  component: RecommendationApisSinglePage,
});

function RecommendationApisSinglePage() {
  return (
    <ApiGroupRunner
      title="Recommendation APIs Single"
      subtitle="Covers all recommendation single endpoints with the same smoke-test payload."
      apiBase={API_BASE}
      defaultDatasetPath="data/processed/vanilla_kpi_dataset.json"
      endpoints={[
        {
          id: "generate-rules-single",
          label: "Generate Rules Single",
          method: "POST",
          path: "/api/generate-rules-single",
          datasetField: "json_path",
        },
        {
          id: "similar-business-recommendations-single",
          label: "Similar Business Recommendations Single",
          method: "POST",
          path: "/api/similar-business-recommendations-single",
          datasetField: "json_path",
        },
        {
          id: "benchmark-dashboard-single",
          label: "Benchmark Dashboard Single",
          method: "POST",
          path: "/api/benchmark-dashboard-single",
          datasetField: "json_path",
        },
        {
          id: "next-post-recommendations-single",
          label: "Next Post Recommendations Single",
          method: "POST",
          path: "/api/next-post-recommendations-single",
          datasetField: "json_path",
        },
      ]}
    />
  );
}
