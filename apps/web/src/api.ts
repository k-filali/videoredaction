import type {
  AuditLog,
  ClassTreatment,
  ExportAccepted,
  ExportArtifact,
  ModelRun,
  NormalizedBox,
  RedactionStyle,
  ReprocessingSuggestion,
  ReprocessingSuggestionResolution,
  ReviewCommand,
  ReviewMutation,
  ReviewSnapshot,
  ProcessingAccepted,
  ResumableUploadInitiated,
  UploadAccepted,
  UploadCapability,
  VideoAsset,
  VideoListResponse,
  VideoStatusResponse,
} from "./types";
import {
  RESUMABLE_CHUNK_SIZE_BYTES,
  uploadFileResumable,
} from "./lib/resumableUpload";

const REVIEWER_SESSION_KEY = "clearframe-reviewer-session";

function getReviewerSession(): string {
  const existing = sessionStorage.getItem(REVIEWER_SESSION_KEY);
  if (existing) return existing;
  const generated = `review-${crypto.randomUUID().slice(0, 12)}`;
  sessionStorage.setItem(REVIEWER_SESSION_KEY, generated);
  return generated;
}

export class ApiError extends Error {
  status: number;
  detail: unknown;

  constructor(message: string, status: number, detail?: unknown) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
  }
}

function apiHeaders(includeJson = false): HeadersInit {
  const headers: Record<string, string> = {
    "X-Reviewer-Session": getReviewerSession(),
  };
  if (includeJson) headers["Content-Type"] = "application/json";
  return headers;
}

async function parseError(response: Response): Promise<ApiError> {
  let detail: unknown;
  try {
    const body = (await response.json()) as { detail?: unknown };
    detail = body.detail;
  } catch {
    detail = undefined;
  }
  const nested =
    detail && typeof detail === "object" && "message" in detail
      ? String((detail as { message: unknown }).message)
      : undefined;
  const message =
    nested ??
    (typeof detail === "string" ? detail : `Request failed with status ${response.status}`);
  return new ApiError(message, response.status, detail);
}

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, {
    ...init,
    headers: {
      ...apiHeaders(Boolean(init?.body)),
      ...init?.headers,
    },
  });
  if (!response.ok) throw await parseError(response);
  return (await response.json()) as T;
}

export function mediaUrl(path: string | null): string | null {
  return path;
}

export const api = {
  listVideos: () => request<VideoListResponse>("/api/videos"),

  getUploadCapability: () =>
    request<UploadCapability>("/api/uploads/capability"),

  initiateUpload: (file: File) =>
    request<ResumableUploadInitiated>("/api/uploads/initiate", {
      method: "POST",
      body: JSON.stringify({
        filename: file.name,
        content_type: file.type || "application/octet-stream",
        size_bytes: file.size,
        hashed_case_id: null,
      }),
    }),

  completeUpload: (uploadId: string) =>
    request<UploadAccepted>(
      `/api/uploads/${encodeURIComponent(uploadId)}/complete`,
      {
        method: "POST",
        body: JSON.stringify({}),
      },
    ),

  getVideo: (videoId: string) => request<VideoAsset>(`/api/videos/${videoId}`),

  getVideoStatus: (videoId: string) =>
    request<VideoStatusResponse>(`/api/videos/${videoId}/status`),

  getTracks: (videoId: string) =>
    request<ReviewSnapshot>(`/api/videos/${videoId}/tracks`),

  getAudit: (videoId: string) => request<AuditLog>(`/api/videos/${videoId}/audit`),

  processVideo: (videoId: string, sampleEveryFrames = 5, modelIds?: string[]) =>
    request<ProcessingAccepted>(`/api/videos/${videoId}/process`, {
      method: "POST",
      body: JSON.stringify({
        sample_every_frames: sampleEveryFrames,
        ...(modelIds && modelIds.length > 0 ? { model_ids: modelIds } : {}),
      }),
    }),

  getLatestModelRun: (videoId: string) =>
    request<ModelRun>(`/api/videos/${videoId}/model-runs/latest`),

  getReprocessingSuggestions: (videoId: string) =>
    request<ReprocessingSuggestion[]>(
      `/api/videos/${videoId}/reprocessing-suggestions`,
    ),

  resolveReprocessingSuggestion: (
    videoId: string,
    suggestionId: string,
    decision: "accept" | "dismiss",
    expectedRevision: number,
  ) =>
    request<ReprocessingSuggestionResolution>(
      `/api/videos/${videoId}/reprocessing-suggestions/${suggestionId}/${decision}`,
      {
        method: "POST",
        body: JSON.stringify({
          expected_revision: expectedRevision,
          reason_code:
            decision === "accept"
              ? "reviewer_accepted_context"
              : "reviewer_dismissed_context",
        }),
      },
    ),

  updateTrack: (videoId: string, trackId: string, command: ReviewCommand) =>
    request<ReviewMutation>(`/api/videos/${videoId}/tracks/${trackId}`, {
      method: "PATCH",
      body: JSON.stringify(command),
    }),

  createReviewAction: (videoId: string, command: ReviewCommand) =>
    request<ReviewMutation>(`/api/videos/${videoId}/review-actions`, {
      method: "POST",
      body: JSON.stringify(command),
    }),

  createManualRegion: (
    videoId: string,
    input: {
      expected_revision: number;
      frame_index: number;
      timestamp_ms: number;
      class_name: string;
      bbox: NormalizedBox;
      start_frame?: number;
      end_frame?: number;
      start_ms?: number;
      end_ms?: number;
      reason_code?: string;
    },
  ) =>
    request<ReviewMutation>(`/api/videos/${videoId}/manual-regions`, {
      method: "POST",
      body: JSON.stringify(input),
    }),

  requestExport: (
    videoId: string,
    expectedRevision: number,
    style: RedactionStyle,
    classTreatments: Record<string, ClassTreatment> = {},
  ) =>
    request<ExportAccepted>(`/api/videos/${videoId}/exports`, {
      method: "POST",
      body: JSON.stringify({
        expected_revision: expectedRevision,
        redaction_style: style,
        class_treatments: classTreatments,
      }),
    }),

  getExport: (exportId: string) =>
    request<ExportArtifact>(`/api/exports/${exportId}`),
};

function uploadVideoMultipart(
  file: File,
  onProgress: (progress: number) => void,
): Promise<UploadAccepted> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/api/videos");
    xhr.responseType = "json";
    xhr.setRequestHeader("X-Reviewer-Session", getReviewerSession());
    xhr.upload.addEventListener("progress", (event) => {
      if (event.lengthComputable) onProgress(event.loaded / event.total);
    });
    xhr.addEventListener("load", () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        onProgress(1);
        resolve(xhr.response as UploadAccepted);
        return;
      }
      const detail = (xhr.response as { detail?: unknown } | null)?.detail;
      reject(
        new ApiError(
          typeof detail === "string" ? detail : `Upload failed with status ${xhr.status}`,
          xhr.status,
          detail,
        ),
      );
    });
    xhr.addEventListener("error", () => reject(new ApiError("Could not reach ClearFrame", 0)));
    const form = new FormData();
    form.append("file", file);
    xhr.send(form);
  });
}

export async function uploadVideo(
  file: File,
  onProgress: (progress: number) => void,
): Promise<UploadAccepted> {
  let capability: UploadCapability;
  try {
    capability = await api.getUploadCapability();
  } catch (error) {
    if (
      error instanceof ApiError &&
      (error.status === 404 || error.status === 405)
    ) {
      return uploadVideoMultipart(file, onProgress);
    }
    throw error;
  }
  if (capability.mode !== "resumable") {
    return uploadVideoMultipart(file, onProgress);
  }
  if (capability.chunk_size_bytes !== RESUMABLE_CHUNK_SIZE_BYTES) {
    throw new ApiError("Cloud upload returned an unsupported chunk size", 500);
  }
  if (file.size > capability.max_upload_bytes) {
    throw new ApiError(
      `Video exceeds the ${Math.floor(capability.max_upload_bytes / (1024 * 1024))} MB limit`,
      413,
    );
  }

  const initiated = await api.initiateUpload(file);
  if (initiated.chunk_size_bytes !== RESUMABLE_CHUNK_SIZE_BYTES) {
    throw new ApiError("Cloud upload session returned an unsupported chunk size", 500);
  }
  await uploadFileResumable(file, initiated.session_url, onProgress);
  return api.completeUpload(initiated.upload_id);
}
