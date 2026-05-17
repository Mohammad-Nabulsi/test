let pendingDatasetFile: File | null = null;

const LATEST_CAPTION_ANALYSIS_KEY = "suhail_latest_caption_analysis";

export type CaptionAnalysisResult = {
  dataset_id?: string;
  rows_received?: number;
  rows_cleaned?: number;
  rows_kpi?: number;
  hashtag_stage?: unknown;
  topic_stage?: unknown;
  created_at: string;
};

export function setPendingDatasetFile(file: File) {
  pendingDatasetFile = file;
}

export function getPendingDatasetFile() {
  return pendingDatasetFile;
}

export function consumePendingDatasetFile() {
  const file = pendingDatasetFile;
  pendingDatasetFile = null;
  return file;
}

export function saveLatestCaptionAnalysis(result: Omit<CaptionAnalysisResult, "created_at">) {
  if (typeof window === "undefined") return;
  const payload: CaptionAnalysisResult = {
    dataset_id: result.dataset_id,
    rows_received: result.rows_received,
    rows_cleaned: result.rows_cleaned,
    rows_kpi: result.rows_kpi,
    hashtag_stage: result.hashtag_stage,
    topic_stage: result.topic_stage,
    created_at: new Date().toISOString(),
  };
  localStorage.setItem(LATEST_CAPTION_ANALYSIS_KEY, JSON.stringify(payload));
}

export function getLatestCaptionAnalysis(): CaptionAnalysisResult | null {
  if (typeof window === "undefined") return null;
  const raw = localStorage.getItem(LATEST_CAPTION_ANALYSIS_KEY);
  if (!raw) return null;
  try {
    return JSON.parse(raw) as CaptionAnalysisResult;
  } catch {
    localStorage.removeItem(LATEST_CAPTION_ANALYSIS_KEY);
    return null;
  }
}
