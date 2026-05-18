import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { useEffect } from "react";

export const Route = createFileRoute("/dashboard/caption-tips")({
  component: CaptionTipsRedirect,
});

function CaptionTipsRedirect() {
  const navigate = useNavigate();
  useEffect(() => {
    navigate({ to: "/dashboard/hashtags", replace: true });
  }, [navigate]);
  return null;
}
