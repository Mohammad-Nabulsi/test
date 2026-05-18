import { createFileRoute } from "@tanstack/react-router";
import { ApiGroupRunner } from "@/components/dashboard/ApiGroupRunner";

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8002";

export const Route = createFileRoute("/dashboard/business-momentum-single")({
  component: BusinessMomentumSinglePage,
});

function BusinessMomentumSinglePage() {
  return (
    <ApiGroupRunner
      title="Business Momentum Single"
      subtitle="Runs momentum analyze/status/sector-summary endpoints and renders all returned fields."
      apiBase={API_BASE}
      defaultDatasetPath="data/vanilla.json"
      endpoints={[
        {
          id: "business-momentum-analyze-single",
          label: "Business Momentum Analyze Single",
          method: "POST",
          path: "/api/business-momentum/analyze-single",
          datasetField: "uploaded_file_path",
        },
        {
          id: "business-momentum-status-single",
          label: "Business Momentum Status Single",
          method: "GET",
          path: "/api/business-momentum/status-single",
          query: { business_name: "$business_name" },
        },
        {
          id: "business-momentum-sector-summary-single",
          label: "Business Momentum Sector Summary Single",
          method: "GET",
          path: "/api/business-momentum/sector-summary-single",
        },
      ]}
    />
  );
}
