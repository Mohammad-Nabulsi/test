import { getLatestCaptionAnalysis } from "@/lib/upload-state";

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000";
const DEFAULT_DATASET_ID = "2c44ce37-a606-4389-a2be-55ff8a1f7397";

export type MeBusiness = {
  name: string;
  nameEn: string;
  handle: string;
  category: string;
  location: string;
  followers: number;
  followersGrowth: number;
  posts: number;
  avatarColor: string;
};

export type MeKpi = {
  key: string;
  label: string;
  value: number | string;
  suffix?: string;
  delta: number;
  spark: number[];
  string?: boolean;
};

export type MeResponse = {
  business: MeBusiness;
  kpis: MeKpi[];
};

export function resolveDatasetId(): string {
  if (typeof window === "undefined") return DEFAULT_DATASET_ID;
  const fromQuery = new URLSearchParams(window.location.search).get("dataset_id");
  if (fromQuery) return fromQuery;
  const latest = getLatestCaptionAnalysis();
  if (latest?.dataset_id) return latest.dataset_id;
  return DEFAULT_DATASET_ID;
}

export async function fetchMeData(datasetId: string): Promise<MeResponse> {
  const res = await fetch(`${API_BASE}/api/me/${encodeURIComponent(datasetId)}`);
  if (!res.ok) {
    throw new Error(`ME API failed: ${res.status}`);
  }
  return (await res.json()) as MeResponse;
}

export function getAvatarLetter(name: string): string {
  const trimmed = (name || "").trim();
  if (!trimmed) return "B";
  return trimmed[0]!.toUpperCase();
}

export function formatCompactArabic(value: number): string {
  return new Intl.NumberFormat("ar", { notation: "compact", maximumFractionDigits: 1 }).format(value || 0);
}
