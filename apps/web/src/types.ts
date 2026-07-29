export type VideoStatus =
  | "UPLOADING"
  | "VALIDATING"
  | "PROXYING"
  | "PROCESSING"
  | "READY_FOR_REVIEW"
  | "EXPORTING"
  | "EXPORTED"
  | "FAILED";

export type JobStatus = "QUEUED" | "RUNNING" | "COMPLETED" | "FAILED" | "CANCELLED";
export type JobType = "INGEST" | "PROXY" | "DETECT" | "REPROCESS" | "EXPORT";
export type ExportStatus = "QUEUED" | "RENDERING" | "VERIFYING" | "COMPLETED" | "FAILED";
export type RedactionStyle = "pixelate" | "gaussian_blur" | "black_box";
export type TrackSource = "MODEL" | "MANUAL" | "REPROCESSING";
export type ReprocessingSuggestionStatus = "PENDING" | "ACCEPTED" | "DISMISSED";

export type ReviewActionType =
  | "ACCEPT_PROPOSAL"
  | "RESTORE_TRACK"
  | "REDACT_TRACK"
  | "DELETE_FALSE_POSITIVE"
  | "CREATE_MANUAL_REGION"
  | "RESIZE_REGION"
  | "MOVE_REGION"
  | "SPLIT_TRACK"
  | "MERGE_TRACKS"
  | "EXTEND_TRACK"
  | "TRIM_TRACK"
  | "CHANGE_CLASS"
  | "UNDO"
  | "REDO"
  | "EXPORT_REQUESTED"
  | "EXPORT_COMPLETED";

export interface VideoAsset {
  id: string;
  original_filename: string;
  content_type: string;
  source_type: string;
  duration_ms: number | null;
  fps: number | null;
  width: number | null;
  height: number | null;
  codec: string | null;
  audio_present: boolean | null;
  original_sha256: string | null;
  status: VideoStatus;
  review_revision: number;
  active_model_run_id: string | null;
  proxy_url: string | null;
  thumbnail_url: string | null;
  error_message: string | null;
  created_at: string;
  updated_at: string;
}

export interface ProcessingJob {
  id: string;
  job_type: JobType;
  status: JobStatus;
  progress: number;
  stage: string;
  attempts: number;
  error_message: string | null;
  created_at: string;
  started_at: string | null;
  completed_at: string | null;
}

export interface VideoListResponse {
  items: VideoAsset[];
  total: number;
}

export interface VideoStatusResponse {
  video: VideoAsset;
  jobs: ProcessingJob[];
}

export interface UploadAccepted {
  video: VideoAsset;
  job: ProcessingJob;
}

export type UploadCapability =
  | { mode: "multipart" }
  | {
      mode: "resumable";
      chunk_size_bytes: number;
      max_upload_bytes: number;
    };

export interface ResumableUploadInitiated {
  upload_id: string;
  session_url: string;
  chunk_size_bytes: number;
}

export interface ModelRun {
  id: string;
  video_id: string;
  detector_versions: Record<string, unknown>;
  tracker_name: string;
  tracker_version: string;
  thresholds: Record<string, unknown>;
  config_hash: string;
  device: string;
  status: "QUEUED" | "RUNNING" | "COMPLETED" | "FAILED";
  metrics: Record<string, unknown>;
  started_at: string | null;
  completed_at: string | null;
  created_at: string;
}

export interface ProcessingAccepted {
  run: ModelRun;
  job: ProcessingJob;
}

export interface NormalizedBox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

export interface ReviewKeyframe {
  frame_index: number;
  timestamp_ms: number;
  bbox: NormalizedBox;
  locked: boolean;
}

export interface ReviewTrack {
  track_id: string;
  class_name: string;
  source: TrackSource;
  active: boolean;
  redacted: boolean;
  accepted: boolean;
  start_frame: number;
  end_frame: number;
  start_ms: number;
  end_ms: number;
  confidence: number | null;
  warning: string | null;
  keyframes: ReviewKeyframe[];
}

export interface ReviewSnapshot {
  video_id: string;
  revision: number;
  tracks: Record<string, ReviewTrack>;
}

export interface ReviewCommand {
  action_type: ReviewActionType;
  expected_revision: number;
  track_id?: string;
  frame_index?: number;
  timestamp_ms?: number;
  reason_code?: string;
  payload?: Record<string, unknown>;
}

export interface AuditAction {
  id: string;
  video_id: string;
  track_id: string | null;
  frame_index: number | null;
  timestamp_ms: number | null;
  action_type: ReviewActionType;
  before_state: Record<string, unknown>;
  after_state: Record<string, unknown>;
  reason_code: string | null;
  reviewer_session_id: string;
  revision: number;
  model_version: string | null;
  application_version: string;
  inverse_of_action_id: string | null;
  created_at: string;
}

export interface ReviewMutation {
  action: AuditAction;
  state: ReviewSnapshot;
  reprocessing_job: ProcessingJob | null;
  reprocessing_note: string | null;
}

export interface AuditLog {
  video_id: string;
  revision: number;
  actions: AuditAction[];
}

export interface ReprocessingSuggestion {
  id: string;
  source_action_id: string;
  job_id: string;
  track_id: string;
  source_revision: number;
  class_name: string;
  seed_frame_index: number;
  frame_index: number;
  timestamp_ms: number;
  bbox: NormalizedBox;
  confidence: number;
  direction: string;
  propagation_method: string;
  seed_locked: boolean;
  status: ReprocessingSuggestionStatus;
  metadata: Record<string, unknown>;
  created_at: string;
  resolved_at: string | null;
  resolved_by_session_id: string | null;
  resolution_reason_code: string | null;
  resolution_action_id: string | null;
}

export interface ReprocessingSuggestionResolution {
  suggestion: ReprocessingSuggestion;
  state: ReviewSnapshot;
  action: AuditAction | null;
}

export interface ExportArtifact {
  id: string;
  video_id: string;
  status: ExportStatus;
  redaction_style: RedactionStyle;
  source_model_run_id: string | null;
  review_revision: number;
  export_sha256: string | null;
  download_url: string | null;
  manifest_url: string | null;
  error_message: string | null;
  created_at: string;
  completed_at: string | null;
}

export interface ExportAccepted {
  export: ExportArtifact;
  job: ProcessingJob;
}
