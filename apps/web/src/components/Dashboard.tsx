import { useCallback, useEffect, useRef, useState } from "react";

import { ApiError, api, mediaUrl, uploadVideo } from "../api";
import { formatTime, humanFileSize } from "../lib/geometry";
import type { VideoAsset } from "../types";
import { Brand, PrivacyStatus } from "./Brand";
import { Icon } from "./Icon";

interface DashboardProps {
  onOpenVideo: (video: VideoAsset) => void;
  onNotify: (message: string, tone?: "success" | "error" | "info") => void;
}

interface ActiveUpload {
  name: string;
  size: number;
  progress: number;
  stage: "uploading" | "processing";
}

const workingStatuses = new Set([
  "UPLOADING",
  "VALIDATING",
  "PROXYING",
  "PROCESSING",
  "EXPORTING",
]);

const statusCopy: Record<VideoAsset["status"], string> = {
  UPLOADING: "Uploading",
  VALIDATING: "Validating",
  PROXYING: "Building proxy",
  PROCESSING: "Detecting regions",
  READY_FOR_REVIEW: "Ready for review",
  EXPORTING: "Exporting",
  EXPORTED: "Exported",
  FAILED: "Needs attention",
};

function formatDate(value: string): string {
  const date = new Date(value);
  const today = new Date();
  if (date.toDateString() === today.toDateString()) {
    return `Today, ${date.toLocaleTimeString([], { hour: "numeric", minute: "2-digit" })}`;
  }
  return date.toLocaleDateString([], { month: "short", day: "numeric", year: "numeric" });
}

function VideoCard({
  video,
  onOpen,
  onDetect,
  detecting,
}: {
  video: VideoAsset;
  onOpen: (video: VideoAsset) => void;
  onDetect: (video: VideoAsset) => void;
  detecting: boolean;
}) {
  const ready =
    video.status === "EXPORTED" ||
    (video.status === "READY_FOR_REVIEW" &&
      (video.active_model_run_id !== null || video.review_revision > 0));
  const needsDetection =
    video.status === "READY_FOR_REVIEW" &&
    video.active_model_run_id === null &&
    video.review_revision === 0;
  const working = workingStatuses.has(video.status) || detecting;
  const progressHint =
    video.status === "VALIDATING"
      ? 24
      : video.status === "PROXYING"
        ? 51
        : video.status === "PROCESSING"
          ? 76
          : video.status === "EXPORTING"
            ? 88
            : 12;

  return (
    <article className="case-card">
      <button
        className="case-thumbnail"
        type="button"
        onClick={() => ready && onOpen(video)}
        disabled={!ready}
        aria-label={ready ? `Open ${video.original_filename}` : `${video.original_filename} is processing`}
      >
        {video.thumbnail_url ? (
          <img src={mediaUrl(video.thumbnail_url) ?? undefined} alt="" />
        ) : (
          <span className="thumbnail-placeholder">
            <Icon name="file-video" size={30} />
          </span>
        )}
        <span className={`status-chip status-${video.status.toLowerCase()}`}>
          {working && <span className="status-pulse" />}
          {detecting
            ? "Starting detection"
            : needsDetection
              ? "Detection needed"
              : statusCopy[video.status]}
        </span>
        {video.duration_ms !== null && (
          <span className="duration-chip">{formatTime(video.duration_ms).slice(0, 5)}</span>
        )}
      </button>

      <div className="case-card-body">
        <div className="case-card-heading">
          <div>
            <h3 title={video.original_filename}>{video.original_filename}</h3>
            <p>{formatDate(video.created_at)}</p>
          </div>
          <button
            className="icon-button subtle"
            type="button"
            aria-label={`More options for ${video.original_filename}`}
          >
            <span className="more-dots">•••</span>
          </button>
        </div>

        {working ? (
          <div className="case-progress">
            <div className="progress-line">
              <span style={{ width: `${progressHint}%` }} />
            </div>
            <span>Preparing a private review copy…</span>
          </div>
        ) : video.status === "FAILED" ? (
          <p className="case-error">{video.error_message ?? "Processing could not be completed."}</p>
        ) : (
          <div className="case-meta">
            <span>{video.width && video.height ? `${video.width} × ${video.height}` : "Video"}</span>
            <i />
            <span>{video.codec?.toUpperCase() ?? "Media ready"}</span>
            <span className="case-revision">Revision {video.review_revision}</span>
          </div>
        )}
      </div>

      {needsDetection ? (
        <button
          className="case-open-button"
          type="button"
          onClick={() => onDetect(video)}
          disabled={detecting}
        >
          {detecting ? "Starting detection" : "Run detection"}
          <span aria-hidden="true">→</span>
        </button>
      ) : ready ? (
        <button className="case-open-button" type="button" onClick={() => onOpen(video)}>
          Open review
          <span aria-hidden="true">→</span>
        </button>
      ) : null}
    </article>
  );
}

export function Dashboard({ onOpenVideo, onNotify }: DashboardProps) {
  const [videos, setVideos] = useState<VideoAsset[]>([]);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [upload, setUpload] = useState<ActiveUpload | null>(null);
  const [dragging, setDragging] = useState(false);
  const [query, setQuery] = useState("");
  const [detectingIds, setDetectingIds] = useState<Set<string>>(new Set());
  const inputRef = useRef<HTMLInputElement>(null);
  const uploadBusyRef = useRef(false);

  const loadVideos = useCallback(async (quiet = false) => {
    if (!quiet) setLoading(true);
    try {
      const response = await api.listVideos();
      setVideos(response.items);
      setLoadError(null);
    } catch (error) {
      if (!quiet) setLoadError(error instanceof Error ? error.message : "Could not load videos");
    } finally {
      if (!quiet) setLoading(false);
    }
  }, []);

  useEffect(() => {
    const timer = window.setTimeout(() => void loadVideos(), 0);
    return () => window.clearTimeout(timer);
  }, [loadVideos]);

  useEffect(() => {
    if (!videos.some((video) => workingStatuses.has(video.status))) return;
    const timer = window.setInterval(() => void loadVideos(true), 1800);
    return () => window.clearInterval(timer);
  }, [loadVideos, videos]);

  const prepareForReview = useCallback(async (videoId: string): Promise<VideoAsset> => {
    const deadline = Date.now() + 10 * 60 * 1000;
    let detectionRequested = false;
    while (Date.now() < deadline) {
      const status = await api.getVideoStatus(videoId);
      const latest = status.video;
      setVideos((current) => [
        latest,
        ...current.filter((item) => item.id !== latest.id),
      ]);
      if (latest.status === "FAILED") {
        throw new Error(latest.error_message ?? "Video processing failed");
      }
      if (
        latest.status === "READY_FOR_REVIEW" &&
        latest.active_model_run_id !== null
      ) {
        return latest;
      }
      if (
        latest.status === "READY_FOR_REVIEW" &&
        latest.active_model_run_id === null &&
        !detectionRequested
      ) {
        detectionRequested = true;
        try {
          await api.processVideo(videoId);
        } catch (error) {
          if (!(error instanceof ApiError) || error.status !== 409) throw error;
        }
      }
      await new Promise((resolve) => window.setTimeout(resolve, 700));
    }
    throw new Error("Processing timed out. The video remains available on this device.");
  }, []);

  const runDetection = useCallback(
    async (video: VideoAsset) => {
      setDetectingIds((current) => new Set(current).add(video.id));
      try {
        await prepareForReview(video.id);
        onNotify("Detection complete. The review queue is ready.", "success");
      } catch (error) {
        onNotify(
          error instanceof Error ? error.message : "Detection could not be completed",
          "error",
        );
      } finally {
        setDetectingIds((current) => {
          const next = new Set(current);
          next.delete(video.id);
          return next;
        });
      }
    },
    [onNotify, prepareForReview],
  );

  const handleFile = useCallback(
    async (file: File) => {
      if (uploadBusyRef.current) {
        onNotify("Finish the current upload before adding another video.", "info");
        return;
      }
      const supported = file.type.startsWith("video/") || /\.(mp4|mov|m4v|avi)$/i.test(file.name);
      if (!supported) {
        onNotify("Choose an MP4, MOV, M4V, or AVI video.", "error");
        return;
      }
      uploadBusyRef.current = true;
      setUpload({ name: file.name, size: file.size, progress: 0, stage: "uploading" });
      try {
        const accepted = await uploadVideo(file, (progress) => {
          setUpload((current) => (current ? { ...current, progress } : current));
        });
        setVideos((current) => [
          accepted.video,
          ...current.filter((item) => item.id !== accepted.video.id),
        ]);
        setUpload((current) => (current ? { ...current, stage: "processing" } : current));
        onNotify("Upload secured. Preparing proxy and detection.", "success");
        await prepareForReview(accepted.video.id);
        setUpload(null);
        onNotify("Detection complete. The review queue is ready.", "success");
      } catch (error) {
        setUpload(null);
        onNotify(error instanceof Error ? error.message : "Upload failed", "error");
      } finally {
        uploadBusyRef.current = false;
      }
    },
    [onNotify, prepareForReview],
  );

  const filteredVideos = videos.filter((video) =>
    video.original_filename.toLowerCase().includes(query.trim().toLowerCase()),
  );

  return (
    <div className="dashboard">
      <header className="dashboard-header">
        <a className="brand-link" href="/" aria-label="ClearFrame home">
          <Brand />
        </a>
        <div className="header-actions">
          <PrivacyStatus />
          <button
            className="button button-primary upload-header-button"
            type="button"
            onClick={() => inputRef.current?.click()}
            disabled={upload !== null}
          >
            <Icon name="upload" size={17} />
            Upload video
          </button>
        </div>
      </header>

      <main className="dashboard-main">
        <section className="dashboard-intro" aria-labelledby="dashboard-title">
          <div>
            <p className="section-kicker">Review workspace</p>
            <h1 id="dashboard-title">Video evidence</h1>
            <p>Review suggested redactions and produce a traceable derived copy.</p>
          </div>
          <div className="research-note">
            <Icon name="info" size={17} />
            <span>
              <strong>Human decision required.</strong> Suggestions are never exported without
              reviewer confirmation.
            </span>
          </div>
        </section>

        <section
          className={`upload-zone${dragging ? " is-dragging" : ""}`}
          onDragEnter={(event) => {
            event.preventDefault();
            setDragging(true);
          }}
          onDragOver={(event) => event.preventDefault()}
          onDragLeave={(event) => {
            if (event.currentTarget === event.target) setDragging(false);
          }}
          onDrop={(event) => {
            event.preventDefault();
            setDragging(false);
            const file = event.dataTransfer.files[0];
            if (file) void handleFile(file);
          }}
        >
          <input
            ref={inputRef}
            className="visually-hidden"
            type="file"
            disabled={upload !== null}
            accept="video/mp4,video/quicktime,video/x-msvideo,.m4v"
            onChange={(event) => {
              const file = event.currentTarget.files?.[0];
              if (file) void handleFile(file);
              event.currentTarget.value = "";
            }}
          />
          {upload ? (
            <div className="active-upload">
              <span className="upload-file-icon">
                {upload.stage === "processing" ? (
                  <span className="spinner" />
                ) : (
                  <Icon name="file-video" size={22} />
                )}
              </span>
              <div className="active-upload-copy">
                <div>
                  <strong>{upload.name}</strong>
                  <span>
                    {upload.stage === "processing"
                      ? "Upload complete · securing and validating"
                      : `${Math.round(upload.progress * 100)}% · ${humanFileSize(upload.size)}`}
                  </span>
                </div>
                <div className="upload-progress-track">
                  <span
                    className={upload.stage === "processing" ? "processing" : ""}
                    style={{ width: `${upload.progress * 100}%` }}
                  />
                </div>
              </div>
            </div>
          ) : (
            <>
              <span className="upload-zone-icon">
                <Icon name="upload" size={23} />
              </span>
              <div>
                <strong>Drop video evidence here</strong>
                <span>
                  or{" "}
                  <button type="button" onClick={() => inputRef.current?.click()}>
                    choose a file
                  </button>{" "}
                  from this device
                </span>
              </div>
              <small>MP4, MOV, M4V or AVI · source file remains unchanged</small>
            </>
          )}
        </section>

        <section className="cases-section" aria-labelledby="recent-title">
          <div className="section-heading-row">
            <div>
              <h2 id="recent-title">Recent videos</h2>
              <span>{videos.length} {videos.length === 1 ? "item" : "items"}</span>
            </div>
            <label className="search-field">
              <Icon name="search" size={17} />
              <span className="visually-hidden">Search videos</span>
              <input
                type="search"
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                placeholder="Search video name"
              />
            </label>
          </div>

          {loading ? (
            <div className="case-grid" aria-label="Loading videos">
              {[0, 1, 2].map((item) => (
                <div className="case-card skeleton-card" key={item}>
                  <span />
                  <i />
                  <i />
                </div>
              ))}
            </div>
          ) : loadError ? (
            <div className="inline-empty-state">
              <span><Icon name="warning" size={21} /></span>
              <h3>Videos could not be loaded</h3>
              <p>{loadError}</p>
              <button className="button button-secondary" type="button" onClick={() => void loadVideos()}>
                Try again
              </button>
            </div>
          ) : filteredVideos.length > 0 ? (
            <div className="case-grid">
              {filteredVideos.map((video) => (
                <VideoCard
                  video={video}
                  onOpen={onOpenVideo}
                  onDetect={(item) => void runDetection(item)}
                  detecting={detectingIds.has(video.id)}
                  key={video.id}
                />
              ))}
            </div>
          ) : (
            <div className="inline-empty-state">
              <span><Icon name={query ? "search" : "file-video"} size={23} /></span>
              <h3>{query ? "No matching videos" : "No video evidence yet"}</h3>
              <p>
                {query
                  ? "Try another filename or clear the search."
                  : "Upload a short clip to begin the review workflow."}
              </p>
              {!query && (
                <button
                  className="button button-secondary"
                  type="button"
                  onClick={() => inputRef.current?.click()}
                >
                  <Icon name="plus" size={16} />
                  Choose video
                </button>
              )}
            </div>
          )}
        </section>
      </main>

      <footer className="dashboard-footer">
        <span>ClearFrame research prototype</span>
        <span>Original evidence is never modified</span>
      </footer>
    </div>
  );
}
