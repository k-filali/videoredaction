import {
  memo,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";

import { ApiError, api, mediaUrl } from "../api";
import { boxAtFrame, formatTime } from "../lib/geometry";
import type {
  AuditAction,
  AuditLog,
  ExportArtifact,
  NormalizedBox,
  RedactionStyle,
  ReprocessingSuggestion,
  ReviewActionType,
  ReviewSnapshot,
  ReviewTrack,
  VideoAsset,
} from "../types";
import { AuditDrawer } from "./AuditDrawer";
import { Brand, PrivacyStatus } from "./Brand";
import { ExportModal } from "./ExportModal";
import { Icon } from "./Icon";
import { Timeline } from "./Timeline";
import { VideoCanvas, type VideoCanvasHandle } from "./VideoCanvas";

interface ReviewWorkspaceProps {
  videoId: string;
  onBack: () => void;
  onNotify: (message: string, tone?: "success" | "error" | "info") => void;
}

type ReviewFilter =
  | "all"
  | "needs-review"
  | "reviewed"
  | "restored"
  | "low-confidence"
  | "warnings";

const classColours: Record<string, string> = {
  license_plate: "#61d6b0",
  face: "#b39cff",
  scene_text: "#f2b95d",
};

const exportFinished = new Set(["COMPLETED", "FAILED"]);

const detectionKitOptions = [
  { id: "yolov9t-plate", label: "License plates" },
  { id: "yunet-face", label: "Faces" },
];

function classLabel(value: string): string {
  if (value === "license_plate") return "License plate";
  return value.charAt(0).toUpperCase() + value.slice(1).replaceAll("_", " ");
}

function trackState(track: ReviewTrack): {
  label: string;
  tone: "pending" | "reviewed" | "restored" | "removed";
} {
  if (!track.active) return { label: "Removed", tone: "removed" };
  if (!track.redacted) return { label: "Restored", tone: "restored" };
  if (!track.accepted) return { label: "Needs review", tone: "pending" };
  return { label: "Redacted", tone: "reviewed" };
}

function matchesFilter(track: ReviewTrack, filter: ReviewFilter): boolean {
  if (filter === "needs-review") return !track.accepted;
  if (filter === "reviewed") return track.accepted && track.active && track.redacted;
  if (filter === "restored") return track.accepted && (!track.redacted || !track.active);
  if (filter === "low-confidence") {
    return track.confidence !== null && track.confidence < 0.8;
  }
  if (filter === "warnings") return Boolean(track.warning);
  return true;
}

function compactHash(hash: string | null): string {
  if (!hash) return "Pending";
  return `${hash.slice(0, 8)}…${hash.slice(-6)}`;
}

function historyStacks(log: AuditLog): { undo: string[]; redo: string[] } {
  const undo: string[] = [];
  const redo: string[] = [];
  log.actions.forEach((action) => {
    if (action.action_type === "UNDO" && action.inverse_of_action_id) {
      const index = undo.lastIndexOf(action.inverse_of_action_id);
      if (index >= 0) undo.splice(index, 1);
      redo.push(action.inverse_of_action_id);
    } else if (action.action_type === "REDO" && action.inverse_of_action_id) {
      undo.push(action.inverse_of_action_id);
      const index = redo.lastIndexOf(action.inverse_of_action_id);
      if (index >= 0) redo.splice(index, 1);
    } else if (
      action.action_type !== "EXPORT_REQUESTED" &&
      action.action_type !== "EXPORT_COMPLETED"
    ) {
      undo.push(action.id);
      redo.length = 0;
    }
  });
  return { undo, redo };
}

const trackRowHeight = 74;
const trackListOverscan = 5;

interface VirtualTrackListProps {
  tracks: ReviewTrack[];
  currentFrame: number;
  selectedTrackId: string | null;
  onSelectTrack: (trackId: string) => void;
}

const VirtualTrackList = memo(function VirtualTrackList({
  tracks,
  currentFrame,
  selectedTrackId,
  onSelectTrack,
}: VirtualTrackListProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const scrollTopRef = useRef(0);
  const [scrollTop, setScrollTop] = useState(0);
  const [viewportHeight, setViewportHeight] = useState(600);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    const updateHeight = () =>
      setViewportHeight((height) =>
        height === container.clientHeight ? height : container.clientHeight,
      );
    updateHeight();
    if (typeof ResizeObserver === "undefined") return;
    const observer = new ResizeObserver(updateHeight);
    observer.observe(container);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    const container = containerRef.current;
    const selectedIndex = tracks.findIndex(
      (track) => track.track_id === selectedTrackId,
    );
    if (!container || selectedIndex < 0) return;
    const top = selectedIndex * trackRowHeight;
    const bottom = top + trackRowHeight;
    const currentTop = scrollTopRef.current;
    if (top >= currentTop && bottom <= currentTop + container.clientHeight) return;
    const nextTop = Math.max(
      0,
      top - Math.max(0, (container.clientHeight - trackRowHeight) / 2),
    );
    container.scrollTop = nextTop;
    scrollTopRef.current = nextTop;
    setScrollTop(nextTop);
  }, [selectedTrackId, tracks]);

  const startIndex = Math.max(
    0,
    Math.floor(scrollTop / trackRowHeight) - trackListOverscan,
  );
  const endIndex = Math.min(
    tracks.length,
    Math.ceil((scrollTop + viewportHeight) / trackRowHeight) +
      trackListOverscan,
  );

  return (
    <div
      className="track-list"
      ref={containerRef}
      onScroll={(event) => {
        const nextTop = event.currentTarget.scrollTop;
        scrollTopRef.current = nextTop;
        setScrollTop(nextTop);
      }}
    >
      <div
        className="track-list-virtual"
        style={{ height: `${tracks.length * trackRowHeight}px` }}
      >
        {tracks.slice(startIndex, endIndex).map((track, offset) => {
          const index = startIndex + offset;
          const state = trackState(track);
          const present = boxAtFrame(track, currentFrame) !== null;
          return (
            <button
              className={`track-card track-card-virtual${
                track.track_id === selectedTrackId ? " is-selected" : ""
              }${present ? " is-present" : ""}`}
              type="button"
              key={track.track_id}
              style={{ transform: `translateY(${index * trackRowHeight}px)` }}
              onClick={() => onSelectTrack(track.track_id)}
              aria-setsize={tracks.length}
              aria-posinset={index + 1}
            >
              <span
                className="track-colour"
                style={{
                  background: classColours[track.class_name] ?? "#8ca9a1",
                }}
              />
              <span className="track-card-main">
                <span className="track-card-title">
                  <strong>
                    {classLabel(track.class_name)} {index + 1}
                  </strong>
                  <i className={`track-state state-${state.tone}`}>
                    {state.label}
                  </i>
                </span>
                <span className="track-card-meta">
                  {formatTime(track.start_ms)} – {formatTime(track.end_ms)}
                  {track.confidence !== null && (
                    <i>{Math.round(track.confidence * 100)}%</i>
                  )}
                  {track.source === "MANUAL" && <i>Manual</i>}
                </span>
                {track.warning && (
                  <span className="track-warning">
                    <Icon name="warning" size={13} />
                    {track.warning}
                  </span>
                )}
              </span>
            </button>
          );
        })}
      </div>
    </div>
  );
});

export function ReviewWorkspace({ videoId, onBack, onNotify }: ReviewWorkspaceProps) {
  const playerRef = useRef<VideoCanvasHandle>(null);
  const currentTimeRef = useRef(0);
  const currentFrameRef = useRef(0);
  const playingRef = useRef(false);
  const draftBoxRef = useRef<NormalizedBox | null>(null);
  const lastPlaybackRenderRef = useRef(0);
  const [video, setVideo] = useState<VideoAsset | null>(null);
  const [snapshot, setSnapshot] = useState<ReviewSnapshot | null>(null);
  const [audit, setAudit] = useState<AuditLog | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [auditLoading, setAuditLoading] = useState(false);
  const [selectedTrackId, setSelectedTrackId] = useState<string | null>(null);
  const [currentTimeMs, setCurrentTimeMs] = useState(0);
  const [currentFrame, setCurrentFrame] = useState(0);
  const [durationMs, setDurationMs] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [showRedactions, setShowRedactions] = useState(true);
  const [showBoxes, setShowBoxes] = useState(true);
  const [previewStyle, setPreviewStyle] = useState<RedactionStyle>("pixelate");
  const [filter, setFilter] = useState<ReviewFilter>("all");
  const [classFilters, setClassFilters] = useState<Set<string>>(new Set());
  const [filterMenuOpen, setFilterMenuOpen] = useState(false);
  const [manualMode, setManualMode] = useState(false);
  const [manualClass, setManualClass] = useState("license_plate");
  const [manualStartMs, setManualStartMs] = useState(0);
  const [manualEndMs, setManualEndMs] = useState(0);
  const [draftBox, setDraftBox] = useState<NormalizedBox | null>(null);
  const [mutationBusy, setMutationBusy] = useState(false);
  const [detectionBusy, setDetectionBusy] = useState(false);
  const [suggestions, setSuggestions] = useState<ReprocessingSuggestion[]>([]);
  const [reprocessingStatus, setReprocessingStatus] = useState<string | null>(null);
  const [spanStartMs, setSpanStartMs] = useState(0);
  const [spanEndMs, setSpanEndMs] = useState(0);
  const [auditOpen, setAuditOpen] = useState(false);
  const [exportOpen, setExportOpen] = useState(false);
  const [exportWarnings, setExportWarnings] = useState<string[]>([]);
  const [detectionKits, setDetectionKits] = useState<Set<string>>(
    new Set(detectionKitOptions.map((option) => option.id)),
  );
  const [exporting, setExporting] = useState(false);
  const [exportPollError, setExportPollError] = useState<string | null>(null);
  const [artifact, setArtifact] = useState<ExportArtifact | null>(null);
  const [undoStack, setUndoStack] = useState<string[]>([]);
  const [redoStack, setRedoStack] = useState<string[]>([]);

  useEffect(() => {
    draftBoxRef.current = draftBox;
  }, [draftBox]);

  const applySnapshot = useCallback(
    (result: ReviewSnapshot, preferredTrackId?: string | null) => {
      const candidateId =
        preferredTrackId === undefined ? selectedTrackId : preferredTrackId;
      const track =
        (candidateId ? result.tracks[candidateId] : undefined) ??
        Object.values(result.tracks).find((item) => !item.accepted) ??
        Object.values(result.tracks)[0];
      setSnapshot(result);
      setSelectedTrackId(track?.track_id ?? null);
      setSpanStartMs(track?.start_ms ?? 0);
      setSpanEndMs(track?.end_ms ?? 0);
      return result;
    },
    [selectedTrackId],
  );

  const loadAudit = useCallback(async () => {
    setAuditLoading(true);
    try {
      const result = await api.getAudit(videoId);
      setAudit(result);
      const stacks = historyStacks(result);
      setUndoStack(stacks.undo);
      setRedoStack(stacks.redo);
    } catch (auditError) {
      if (auditOpen) {
        onNotify(auditError instanceof Error ? auditError.message : "Could not load history", "error");
      }
    } finally {
      setAuditLoading(false);
    }
  }, [auditOpen, onNotify, videoId]);

  const reloadSnapshot = useCallback(async () => {
    const result = await api.getTracks(videoId);
    return applySnapshot(result);
  }, [applySnapshot, videoId]);

  const followReprocessing = useCallback(
    async (actionId: string, jobId: string) => {
      setReprocessingStatus("Checking nearby frames without changing reviewer decisions…");
      try {
        for (let attempt = 0; attempt < 30; attempt += 1) {
          await new Promise((resolve) => window.setTimeout(resolve, 600));
          const [items, status] = await Promise.all([
            api.getReprocessingSuggestions(videoId),
            api.getVideoStatus(videoId),
          ]);
          setSuggestions(items);
          const job = status.jobs.find((candidate) => candidate.id === jobId);
          const related = items.filter((item) => item.source_action_id === actionId);
          if (job?.status === "COMPLETED") {
            setReprocessingStatus(
              related.length > 0
                ? `${related.length} context suggestion${related.length === 1 ? "" : "s"} available. Reviewer truth is unchanged.`
                : "Context check completed with no additional suggestions.",
            );
            return;
          }
          if (job?.status === "FAILED" || job?.status === "CANCELLED") {
            setReprocessingStatus("Context checking was unavailable for this edit.");
            return;
          }
        }
        setReprocessingStatus("Context checking is still running in the background.");
      } catch {
        setReprocessingStatus("Context checking is still running in the background.");
      }
    },
    [videoId],
  );

  useEffect(() => {
    let active = true;
    Promise.all([
      api.getVideo(videoId),
      api.getTracks(videoId),
      api.getAudit(videoId),
      api.getReprocessingSuggestions(videoId).catch(() => []),
    ])
      .then(([videoResult, snapshotResult, auditResult, suggestionResult]) => {
        if (!active) return;
        setVideo(videoResult);
        setSnapshot(snapshotResult);
        setAudit(auditResult);
        const stacks = historyStacks(auditResult);
        setUndoStack(stacks.undo);
        setRedoStack(stacks.redo);
        setSuggestions(suggestionResult);
        setDurationMs(videoResult.duration_ms ?? 0);
        const firstPending =
          Object.values(snapshotResult.tracks).find((track) => !track.accepted) ??
          Object.values(snapshotResult.tracks)[0];
        setSelectedTrackId(firstPending?.track_id ?? null);
        setSpanStartMs(firstPending?.start_ms ?? 0);
        setSpanEndMs(firstPending?.end_ms ?? 0);
        setError(null);
      })
      .catch((loadError) => {
        if (active) setError(loadError instanceof Error ? loadError.message : "Review could not be loaded");
      })
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => {
      active = false;
    };
  }, [videoId]);

  const tracks = useMemo(() => (snapshot ? Object.values(snapshot.tracks) : []), [snapshot]);
  const classes = useMemo(
    () => [...new Set(tracks.map((track) => track.class_name))].sort(),
    [tracks],
  );
  const filteredTracks = useMemo(
    () =>
      tracks.filter(
        (track) =>
          matchesFilter(track, filter) &&
          (classFilters.size === 0 || classFilters.has(track.class_name)),
      ),
    [classFilters, filter, tracks],
  );
  const overlayTrackIds = useMemo(
    () => new Set(filteredTracks.map((track) => track.track_id)),
    [filteredTracks],
  );
  const selectedTrack =
    tracks.find((track) => track.track_id === selectedTrackId) ?? null;
  const unresolvedCount = tracks.filter((track) => !track.accepted).length;
  const pendingContextCount = suggestions.filter(
    (item) => item.status === "PENDING",
  ).length;
  const activeRedactions = tracks.filter((track) => track.active && track.redacted).length;
  const selectedSuggestions = selectedTrack
    ? suggestions.filter((item) => item.track_id === selectedTrack.track_id)
    : [];
  const pendingSelectedSuggestions = selectedSuggestions.filter(
    (item) => item.status === "PENDING",
  );

  const selectTrack = useCallback(
    (trackId: string) => {
      const track = snapshot?.tracks[trackId];
      setSelectedTrackId(trackId);
      setDraftBox(null);
      if (track) {
        setSpanStartMs(track.start_ms);
        setSpanEndMs(track.end_ms);
      }
    },
    [snapshot],
  );

  useEffect(() => {
    if (
      selectedTrack &&
      (currentTimeRef.current < selectedTrack.start_ms ||
        currentTimeRef.current > selectedTrack.end_ms)
    ) {
      playerRef.current?.seekToMs(selectedTrack.start_ms);
    }
  }, [selectedTrack]);

  const handleMutationFailure = useCallback(
    async (mutationError: unknown) => {
      if (mutationError instanceof ApiError && mutationError.status === 409) {
        await reloadSnapshot().catch(() => undefined);
        onNotify("Review changed elsewhere. The latest revision is now loaded.", "info");
      } else {
        onNotify(
          mutationError instanceof Error ? mutationError.message : "Review action failed",
          "error",
        );
      }
    },
    [onNotify, reloadSnapshot],
  );

  const updateTrack = useCallback(
    async (
      trackId: string,
      actionType: ReviewActionType,
      options?: {
        payload?: Record<string, unknown>;
        frameIndex?: number;
        timestampMs?: number;
        reasonCode?: string;
      },
    ) => {
      if (!snapshot || mutationBusy) return;
      setMutationBusy(true);
      try {
        const result = await api.updateTrack(videoId, trackId, {
          action_type: actionType,
          expected_revision: snapshot.revision,
          frame_index: options?.frameIndex,
          timestamp_ms: options?.timestampMs,
          reason_code: options?.reasonCode,
          payload: options?.payload ?? {},
        });
        applySnapshot(result.state, trackId);
        setUndoStack((stack) => [...stack, result.action.id]);
        setRedoStack([]);
        setAudit((current) =>
          current
            ? {
                ...current,
                revision: result.state.revision,
                actions: [...current.actions, result.action],
              }
            : current,
        );
        if (result.reprocessing_note) {
          setReprocessingStatus(result.reprocessing_note);
        } else if (result.reprocessing_job) {
          void followReprocessing(result.action.id, result.reprocessing_job.id);
        }
      } catch (mutationError) {
        await handleMutationFailure(mutationError);
      } finally {
        setMutationBusy(false);
      }
    },
    [
      followReprocessing,
      handleMutationFailure,
      mutationBusy,
      applySnapshot,
      snapshot,
      videoId,
    ],
  );

  const createManualRegion = useCallback(
    async (box: NormalizedBox) => {
      if (!snapshot || !video || mutationBusy) return;
      playerRef.current?.pause();
      setMutationBusy(true);
      try {
        const fps = video.fps ?? 30;
        const lastFrame = Math.max(
          0,
          Math.ceil((Math.max(1, durationMs) * fps) / 1000) - 1,
        );
        const anchorFrame = Math.min(lastFrame, currentFrameRef.current);
        const timestampMs = Math.round(currentTimeRef.current);
        const startMs = Math.max(0, Math.min(timestampMs, Math.round(manualStartMs)));
        const endMs = Math.min(
          Math.max(1, durationMs),
          Math.max(timestampMs, Math.round(manualEndMs)),
        );
        const result = await api.createManualRegion(videoId, {
          expected_revision: snapshot.revision,
          frame_index: anchorFrame,
          timestamp_ms: timestampMs,
          class_name: manualClass,
          bbox: box,
          start_frame: Math.min(
            anchorFrame,
            Math.max(0, Math.floor((startMs * fps) / 1000)),
          ),
          end_frame: Math.min(
            lastFrame,
            Math.max(anchorFrame, Math.floor((endMs * fps) / 1000)),
          ),
          start_ms: startMs,
          end_ms: endMs,
          reason_code: "reviewer_identified",
        });
        applySnapshot(result.state, result.action.track_id);
        setUndoStack((stack) => [...stack, result.action.id]);
        setRedoStack([]);
        setAudit((current) =>
          current
            ? {
                ...current,
                revision: result.state.revision,
                actions: [...current.actions, result.action],
              }
            : current,
        );
        setManualMode(false);
        setDraftBox(null);
        onNotify("Manual redaction interval added.", "success");
        if (result.reprocessing_note) {
          setReprocessingStatus(result.reprocessing_note);
        } else if (result.reprocessing_job) {
          void followReprocessing(result.action.id, result.reprocessing_job.id);
        }
      } catch (mutationError) {
        await handleMutationFailure(mutationError);
      } finally {
        setMutationBusy(false);
      }
    },
    [
      durationMs,
      followReprocessing,
      handleMutationFailure,
      applySnapshot,
      manualClass,
      manualEndMs,
      manualStartMs,
      mutationBusy,
      onNotify,
      snapshot,
      video,
      videoId,
    ],
  );

  const undo = useCallback(async () => {
    if (!snapshot || undoStack.length === 0 || mutationBusy) return;
    const actionId = undoStack[undoStack.length - 1]!;
    setMutationBusy(true);
    try {
      const result = await api.createReviewAction(videoId, {
        action_type: "UNDO",
        expected_revision: snapshot.revision,
        payload: { action_id: actionId },
      });
      applySnapshot(result.state);
      setUndoStack((stack) => stack.slice(0, -1));
      setRedoStack((stack) => [...stack, actionId]);
      setAudit((current) =>
        current
          ? { ...current, revision: result.state.revision, actions: [...current.actions, result.action] }
          : current,
      );
      onNotify("Last review change undone.", "info");
    } catch (mutationError) {
      await handleMutationFailure(mutationError);
    } finally {
      setMutationBusy(false);
    }
  }, [
    applySnapshot,
    handleMutationFailure,
    mutationBusy,
    onNotify,
    snapshot,
    undoStack,
    videoId,
  ]);

  const redo = useCallback(async () => {
    if (!snapshot || redoStack.length === 0 || mutationBusy) return;
    const actionId = redoStack[redoStack.length - 1]!;
    setMutationBusy(true);
    try {
      const result = await api.createReviewAction(videoId, {
        action_type: "REDO",
        expected_revision: snapshot.revision,
        payload: { action_id: actionId },
      });
      applySnapshot(result.state);
      setRedoStack((stack) => stack.slice(0, -1));
      setUndoStack((stack) => [...stack, actionId]);
      setAudit((current) =>
        current
          ? { ...current, revision: result.state.revision, actions: [...current.actions, result.action] }
          : current,
      );
      onNotify("Review change reapplied.", "info");
    } catch (mutationError) {
      await handleMutationFailure(mutationError);
    } finally {
      setMutationBusy(false);
    }
  }, [
    applySnapshot,
    handleMutationFailure,
    mutationBusy,
    onNotify,
    redoStack,
    snapshot,
    videoId,
  ]);

  const acceptVisible = useCallback(async () => {
    if (!snapshot || mutationBusy) return;
    const scopedClasses = [...classFilters];
    const pending = tracks.filter(
      (track) =>
        !track.accepted &&
        (scopedClasses.length === 0 || classFilters.has(track.class_name)),
    );
    if (pending.length === 0) {
      onNotify("Every proposal in this scope already has a decision.", "info");
      return;
    }
    setMutationBusy(true);
    let current = snapshot;
    const newActions: AuditAction[] = [];
    try {
      const scopes: (string | null)[] =
        scopedClasses.length === 0 ? [null] : scopedClasses;
      for (const className of scopes) {
        const result = await api.createReviewAction(videoId, {
          action_type: "BULK_ACCEPT",
          expected_revision: current.revision,
          reason_code: "reviewer_bulk_accept",
          payload: className ? { class_name: className } : {},
        });
        current = result.state;
        newActions.push(result.action);
      }
      applySnapshot(current);
      setUndoStack((stack) => [...stack, ...newActions.map((action) => action.id)]);
      setRedoStack([]);
      setAudit((existing) =>
        existing
          ? { ...existing, revision: current.revision, actions: [...existing.actions, ...newActions] }
          : existing,
      );
      onNotify(
        `${pending.length} ${pending.length === 1 ? "track" : "tracks"} accepted in one auditable action.`,
        "success",
      );
    } catch (mutationError) {
      await handleMutationFailure(mutationError);
    } finally {
      setMutationBusy(false);
    }
  }, [
    applySnapshot,
    classFilters,
    handleMutationFailure,
    mutationBusy,
    onNotify,
    snapshot,
    tracks,
    videoId,
  ]);

  const resolveContextSuggestions = useCallback(
    async (decision: "accept" | "dismiss", suggestionId?: string) => {
      if (!snapshot || !selectedTrack || mutationBusy) return;
      const pending = suggestions.filter(
        (item) =>
          item.track_id === selectedTrack.track_id &&
          item.status === "PENDING" &&
          (suggestionId === undefined || item.id === suggestionId),
      );
      if (pending.length === 0) {
        onNotify("No pending context suggestions remain for this track.", "info");
        return;
      }

      setMutationBusy(true);
      let current = snapshot;
      const actions: AuditAction[] = [];
      try {
        for (const suggestion of pending) {
          const result = await api.resolveReprocessingSuggestion(
            videoId,
            suggestion.id,
            decision,
            current.revision,
          );
          current = result.state;
          setSuggestions((items) =>
            items.map((item) =>
              item.id === result.suggestion.id ? result.suggestion : item,
            ),
          );
          if (result.action) actions.push(result.action);
        }

        applySnapshot(current, selectedTrack.track_id);
        if (actions.length > 0) {
          setUndoStack((stack) => [
            ...stack,
            ...actions.map((action) => action.id),
          ]);
          setRedoStack([]);
          setAudit((existing) =>
            existing
              ? {
                  ...existing,
                  revision: current.revision,
                  actions: [...existing.actions, ...actions],
                }
              : existing,
          );
        }
        const verb = decision === "accept" ? "accepted" : "dismissed";
        onNotify(
          `${pending.length} context suggestion${
            pending.length === 1 ? "" : "s"
          } ${verb}.`,
          "success",
        );
      } catch (resolutionError) {
        await Promise.all([
          handleMutationFailure(resolutionError),
          api
            .getReprocessingSuggestions(videoId)
            .then(setSuggestions)
            .catch(() => undefined),
        ]);
      } finally {
        setMutationBusy(false);
      }
    },
    [
      applySnapshot,
      handleMutationFailure,
      mutationBusy,
      onNotify,
      selectedTrack,
      snapshot,
      suggestions,
      videoId,
    ],
  );

  const runDetection = useCallback(async () => {
    if (!video || detectionBusy) return;
    setDetectionBusy(true);
    try {
      if (
        video.status === "READY_FOR_REVIEW" &&
        video.active_model_run_id === null &&
        video.review_revision === 0
      ) {
        try {
          await api.processVideo(videoId, 5, [...detectionKits]);
        } catch (requestError) {
          if (!(requestError instanceof ApiError) || requestError.status !== 409) {
            throw requestError;
          }
        }
      }
      while (true) {
        const status = await api.getVideoStatus(videoId);
        setVideo(status.video);
        if (status.video.status === "FAILED") {
          throw new Error(status.video.error_message ?? "Detection failed");
        }
        if (
          status.video.status === "READY_FOR_REVIEW" &&
          status.video.active_model_run_id !== null
        ) {
          const latest = await reloadSnapshot();
          const firstPending =
            Object.values(latest.tracks).find((track) => !track.accepted) ??
            Object.values(latest.tracks)[0];
          setSelectedTrackId(firstPending?.track_id ?? null);
          setSpanStartMs(firstPending?.start_ms ?? 0);
          setSpanEndMs(firstPending?.end_ms ?? 0);
          await loadAudit();
          onNotify(
            firstPending
              ? "Detection complete. Review every suggested region."
              : "Detection completed with no proposals. Add a manual region if needed.",
            "success",
          );
          return;
        }
        await new Promise((resolve) => window.setTimeout(resolve, 700));
      }
    } catch (detectionError) {
      onNotify(
        detectionError instanceof Error ? detectionError.message : "Detection failed",
        "error",
      );
    } finally {
      setDetectionBusy(false);
    }
  }, [
    detectionBusy,
    detectionKits,
    loadAudit,
    onNotify,
    reloadSnapshot,
    video,
    videoId,
  ]);

  const toggleManualMode = useCallback(() => {
    playerRef.current?.pause();
    if (manualMode) {
      setManualMode(false);
      setDraftBox(null);
      return;
    }
    const anchor = Math.round(currentTimeRef.current);
    setManualStartMs(Math.max(0, anchor - 1000));
    setManualEndMs(Math.min(Math.max(1, durationMs), anchor + 1000));
    setManualMode(true);
    setDraftBox(null);
  }, [durationMs, manualMode]);

  const handleDraftBoxChange = useCallback((box: NormalizedBox | null) => {
    draftBoxRef.current = box;
    setDraftBox(box);
  }, []);

  const handleTimeChange = useCallback((timeMs: number, frame: number) => {
    const frameChanged = frame !== currentFrameRef.current;
    currentTimeRef.current = timeMs;
    currentFrameRef.current = frame;
    if (frameChanged && draftBoxRef.current) {
      draftBoxRef.current = null;
      setDraftBox(null);
    }

    const now = performance.now();
    if (
      !playingRef.current ||
      now - lastPlaybackRenderRef.current >= 200
    ) {
      lastPlaybackRenderRef.current = now;
      setCurrentTimeMs(timeMs);
      setCurrentFrame(frame);
    }
  }, []);

  const handlePlayingChange = useCallback((nextPlaying: boolean) => {
    playingRef.current = nextPlaying;
    setPlaying(nextPlaying);
    if (!nextPlaying) {
      setCurrentTimeMs(currentTimeRef.current);
      setCurrentFrame(currentFrameRef.current);
    }
  }, []);

  const applyTrackSpan = useCallback(async () => {
    if (!selectedTrack || !video) return;
    const requestedStart = Math.round(spanStartMs);
    const requestedEnd = Math.round(spanEndMs);
    const startMs = Math.max(0, Math.min(requestedStart, requestedEnd));
    const endMs = Math.min(
      Math.max(1, durationMs),
      Math.max(startMs, requestedEnd),
    );
    if (
      startMs === selectedTrack.start_ms &&
      endMs === selectedTrack.end_ms
    ) {
      onNotify("Track range is unchanged.", "info");
      return;
    }
    const shiftsForward =
      startMs > selectedTrack.start_ms && endMs > selectedTrack.end_ms;
    const shiftsBackward =
      startMs < selectedTrack.start_ms && endMs < selectedTrack.end_ms;
    if (shiftsForward || shiftsBackward) {
      onNotify("Adjust the range inward or outward in one step.", "info");
      return;
    }
    const fps = video.fps ?? 30;
    const lastFrame = Math.max(
      0,
      Math.ceil((Math.max(1, durationMs) * fps) / 1000) - 1,
    );
    const actionType =
      startMs < selectedTrack.start_ms || endMs > selectedTrack.end_ms
        ? "EXTEND_TRACK"
        : "TRIM_TRACK";
    await updateTrack(selectedTrack.track_id, actionType, {
      payload: {
        start_ms: startMs,
        end_ms: endMs,
        start_frame: Math.min(lastFrame, Math.floor((startMs * fps) / 1000)),
        end_frame: Math.min(lastFrame, Math.floor((endMs * fps) / 1000)),
      },
      reasonCode: "reviewer_range_adjustment",
    });
  }, [
    durationMs,
    onNotify,
    selectedTrack,
    spanEndMs,
    spanStartMs,
    updateTrack,
    video,
  ]);

  const selectNextUnresolved = useCallback(() => {
    const unresolved = tracks.filter((track) => !track.accepted);
    if (unresolved.length === 0) {
      onNotify("Every proposal has a reviewer decision.", "info");
      return;
    }
    const currentIndex = unresolved.findIndex(
      (track) => track.track_id === selectedTrackId,
    );
    const next = unresolved[(currentIndex + 1) % unresolved.length];
    if (next) selectTrack(next.track_id);
  }, [onNotify, selectTrack, selectedTrackId, tracks]);

  const startExport = useCallback(async () => {
    if (!snapshot || exporting) return;
    setExportPollError(null);
    setExporting(true);
    let accepted: Awaited<ReturnType<typeof api.requestExport>>;
    try {
      accepted = await api.requestExport(videoId, snapshot.revision, previewStyle);
    } catch (exportError) {
      setExporting(false);
      await handleMutationFailure(exportError);
      return;
    }
    setArtifact(accepted.export);
    setExportWarnings(accepted.warnings ?? []);
    onNotify("Export started from a frozen review revision.", "success");
    void reloadSnapshot().catch(() => {
      onNotify(
        "Export is running, but the review revision could not be refreshed yet.",
        "info",
      );
    });
  }, [
    exporting,
    handleMutationFailure,
    onNotify,
    previewStyle,
    reloadSnapshot,
    snapshot,
    videoId,
  ]);

  useEffect(() => {
    if (!artifact || exportFinished.has(artifact.status) || !exporting) {
      return;
    }
    const timer = window.setInterval(() => {
      void api
        .getExport(artifact.id)
        .then((updated) => {
          setExportPollError(null);
          setArtifact(updated);
          if (exportFinished.has(updated.status)) {
            setExporting(false);
            if (updated.status === "COMPLETED") {
              onNotify("Export rendered and verified.", "success");
              void Promise.all([api.getVideo(videoId), reloadSnapshot(), loadAudit()])
                .then(([updatedVideo]) => setVideo(updatedVideo))
                .catch(() => {
                  onNotify(
                    "Export is complete, but the workspace could not be refreshed.",
                    "info",
                  );
                });
            }
          }
        })
        .catch(() => {
          setExportPollError(
            "Export status could not be refreshed. The renderer may still be running.",
          );
          setExporting(false);
        });
    }, 1100);
    return () => window.clearInterval(timer);
  }, [artifact, exporting, loadAudit, onNotify, reloadSnapshot, videoId]);

  const retryExportPolling = useCallback(() => {
    if (!artifact || exportFinished.has(artifact.status)) return;
    setExportPollError(null);
    setExporting(true);
  }, [artifact]);

  useEffect(() => {
    const handleKeyboard = (event: KeyboardEvent) => {
      const target = event.target;
      if (
        (target instanceof HTMLElement &&
          (target.isContentEditable ||
            target.closest(
              "input, select, textarea, button, a, [role='button'], [contenteditable='true']",
            ))) ||
        auditOpen ||
        exportOpen
      ) {
        return;
      }
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "z") {
        event.preventDefault();
        if (event.shiftKey) void redo();
        else void undo();
        return;
      }
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "y") {
        event.preventDefault();
        void redo();
        return;
      }
      if (event.code === "Space") {
        event.preventDefault();
        playerRef.current?.togglePlayback();
      } else if (event.key === "ArrowLeft" || event.key.toLowerCase() === "j") {
        event.preventDefault();
        playerRef.current?.stepFrames(event.shiftKey ? -10 : -1);
      } else if (event.key === "ArrowRight" || event.key.toLowerCase() === "l") {
        event.preventDefault();
        playerRef.current?.stepFrames(event.shiftKey ? 10 : 1);
      } else if (event.key.toLowerCase() === "a" && selectedTrack && !selectedTrack.accepted) {
        void updateTrack(selectedTrack.track_id, "ACCEPT_PROPOSAL");
      } else if (event.key.toLowerCase() === "r" && selectedTrack) {
        void updateTrack(
          selectedTrack.track_id,
          selectedTrack.redacted ? "RESTORE_TRACK" : "REDACT_TRACK",
        );
      } else if (
        event.key.toLowerCase() === "b" ||
        event.key.toLowerCase() === "m"
      ) {
        toggleManualMode();
      } else if (event.key.toLowerCase() === "n") {
        selectNextUnresolved();
      } else if (event.key === "Delete" && selectedTrack) {
        void updateTrack(selectedTrack.track_id, "DELETE_FALSE_POSITIVE");
      } else if (event.key === "Escape" && manualMode) {
        setManualMode(false);
        setDraftBox(null);
      }
    };
    window.addEventListener("keydown", handleKeyboard);
    return () => window.removeEventListener("keydown", handleKeyboard);
  }, [
    auditOpen,
    exportOpen,
    manualMode,
    redo,
    selectNextUnresolved,
    selectedTrack,
    toggleManualMode,
    undo,
    updateTrack,
  ]);

  if (loading) {
    return (
      <div className="review-loading">
        <Brand />
        <span className="spinner large" />
        <strong>Opening secure review workspace</strong>
        <p>Loading proxy, tracks, and review history…</p>
      </div>
    );
  }

  if (error || !video || !snapshot || !video.proxy_url) {
    return (
      <div className="review-error">
        <span><Icon name="warning" size={28} /></span>
        <h1>Review workspace unavailable</h1>
        <p>{error ?? "The review proxy is not ready yet."}</p>
        <button className="button button-secondary" type="button" onClick={onBack}>
          <Icon name="arrow-left" size={16} />
          Return to videos
        </button>
      </div>
    );
  }

  return (
    <div className="review-workspace">
      <header className="review-header">
        <div className="review-header-left">
          <button className="icon-button back-button" type="button" onClick={onBack} aria-label="Back to videos">
            <Icon name="arrow-left" />
          </button>
          <a className="review-brand" href="/" onClick={(event) => { event.preventDefault(); onBack(); }}>
            <Brand />
          </a>
          <span className="header-divider" />
          <div className="evidence-title">
            <strong title={video.original_filename}>{video.original_filename}</strong>
            <span>
              {video.width} × {video.height}
              <i />
              {video.fps ? `${video.fps.toFixed(video.fps % 1 ? 2 : 0)} fps` : "Video"}
              <i />
              r{snapshot.revision}
            </span>
          </div>
        </div>
        <div className="review-header-actions">
          <PrivacyStatus />
          <button
            className="button button-quiet"
            type="button"
            onClick={() => {
              setAuditOpen(true);
              void loadAudit();
            }}
          >
            <Icon name="history" size={16} />
            Audit trail
          </button>
          <button
            className="button button-primary"
            type="button"
            disabled={detectionBusy}
            onClick={() => {
              if (!artifact || exportFinished.has(artifact.status)) {
                setArtifact(null);
                setExportPollError(null);
                setExportWarnings([]);
              }
              setExportOpen(true);
            }}
          >
            Export
            <Icon name="download" size={16} />
          </button>
        </div>
      </header>

      <div className="review-body">
        <aside className="track-sidebar">
          <div className="sidebar-header">
            <div>
              <p className="section-kicker">Detected regions</p>
              <h2>Review queue</h2>
            </div>
            <span className="queue-count">{unresolvedCount}</span>
          </div>

          <div className="review-progress-summary">
            <div>
              <span>{tracks.length - unresolvedCount} of {tracks.length} reviewed</span>
              <strong>{tracks.length === 0 ? 0 : Math.round(((tracks.length - unresolvedCount) / tracks.length) * 100)}%</strong>
            </div>
            <div className="progress-line">
              <span
                style={{
                  width: `${tracks.length === 0 ? 0 : ((tracks.length - unresolvedCount) / tracks.length) * 100}%`,
                }}
              />
            </div>
          </div>

          <div className="filter-row">
            <label className="compact-select">
              <span className="visually-hidden">Track status</span>
              <select value={filter} onChange={(event) => setFilter(event.target.value as ReviewFilter)}>
                <option value="all">All tracks</option>
                <option value="needs-review">Needs review</option>
                <option value="reviewed">Redacted</option>
                <option value="restored">Restored / removed</option>
                <option value="low-confidence">Low confidence (&lt;80%)</option>
                <option value="warnings">Warnings</option>
              </select>
              <Icon name="chevron-down" size={13} />
            </label>
            <div className="filter-menu-wrap">
              <button
                className={`icon-button filter-button${classFilters.size > 0 ? " is-active" : ""}`}
                type="button"
                onClick={() => setFilterMenuOpen((open) => !open)}
                aria-expanded={filterMenuOpen}
                aria-label="Filter by class"
              >
                <Icon name="filter" size={16} />
              </button>
              {filterMenuOpen && (
                <div className="class-filter-menu">
                  <strong>Region class</strong>
                  {classes.map((className) => (
                    <label key={className}>
                      <input
                        type="checkbox"
                        checked={classFilters.has(className)}
                        onChange={() =>
                          setClassFilters((current) => {
                            const next = new Set(current);
                            if (next.has(className)) next.delete(className);
                            else next.add(className);
                            return next;
                          })
                        }
                      />
                      <i style={{ background: classColours[className] ?? "#8ca9a1" }} />
                      {classLabel(className)}
                    </label>
                  ))}
                  {classFilters.size > 0 && (
                    <button type="button" onClick={() => setClassFilters(new Set())}>Clear filters</button>
                  )}
                </div>
              )}
            </div>
          </div>

          {tracks.length === 0 ? (
            <div className="track-list">
              <div className="track-list-empty">
                <Icon name="sparkles" size={22} />
                <strong>
                  {video.active_model_run_id === null
                    ? "Run private detection"
                    : "No proposals found"}
                </strong>
                <p>
                  {video.active_model_run_id === null
                    ? "Analyze the proxy locally, then review every suggestion."
                    : "Inspect the clip and add any missed region manually."}
                </p>
                {video.active_model_run_id === null && video.review_revision === 0 ? (
                  <>
                    <div className="detection-kit-picker">
                      {detectionKitOptions.map((option) => (
                        <label key={option.id}>
                          <input
                            type="checkbox"
                            checked={detectionKits.has(option.id)}
                            disabled={detectionBusy}
                            onChange={() =>
                              setDetectionKits((current) => {
                                const next = new Set(current);
                                if (next.has(option.id)) next.delete(option.id);
                                else next.add(option.id);
                                return next;
                              })
                            }
                          />
                          {option.label}
                        </label>
                      ))}
                    </div>
                    <button
                      type="button"
                      onClick={() => void runDetection()}
                      disabled={detectionBusy || detectionKits.size === 0}
                    >
                      {detectionBusy || video.status === "PROCESSING"
                        ? "Detection running…"
                        : "Run detection"}
                    </button>
                  </>
                ) : (
                  <button type="button" onClick={toggleManualMode}>
                    Add manual region
                  </button>
                )}
              </div>
            </div>
          ) : filteredTracks.length === 0 ? (
            <div className="track-list">
              <div className="track-list-empty">
                <Icon name="filter" size={22} />
                <strong>No tracks in this view</strong>
                <button type="button" onClick={() => { setFilter("all"); setClassFilters(new Set()); }}>
                  Clear filters
                </button>
              </div>
            </div>
          ) : (
            <VirtualTrackList
              tracks={filteredTracks}
              currentFrame={currentFrame}
              selectedTrackId={selectedTrackId}
              onSelectTrack={selectTrack}
            />
          )}

          <div className="sidebar-footer">
            <button
              className="button button-ghost"
              type="button"
              onClick={selectNextUnresolved}
              disabled={mutationBusy || unresolvedCount === 0}
            >
              <Icon name="arrow-right" size={15} />
              Next unresolved
            </button>
            <button
              className="button button-secondary"
              type="button"
              onClick={() => void acceptVisible()}
              disabled={mutationBusy}
              title={
                classFilters.size > 0
                  ? "Accept every unconfirmed track in the filtered classes as one auditable action"
                  : "Accept every unconfirmed track as one auditable action"
              }
            >
              <Icon name="check" size={15} />
              {classFilters.size > 0 ? "Accept filtered" : "Accept remaining"}
            </button>
          </div>
        </aside>

        <main className="review-main">
          <div className="viewer-toolbar">
            <div className="preview-toggle-group">
              <button
                className={showRedactions ? "is-active" : ""}
                type="button"
                onClick={() => setShowRedactions(true)}
              >
                Redacted preview
              </button>
              <button
                className={!showRedactions ? "is-active source-warning" : ""}
                type="button"
                onClick={() => setShowRedactions(false)}
              >
                Unredacted proxy
              </button>
            </div>
            <div className="viewer-toolbar-actions">
              <label className="style-select">
                <span>Preview style</span>
                <select value={previewStyle} onChange={(event) => setPreviewStyle(event.target.value as RedactionStyle)}>
                  <option value="pixelate">Pixelate</option>
                  <option value="gaussian_blur">Gaussian blur</option>
                  <option value="black_box">Black box</option>
                </select>
                <Icon name="chevron-down" size={12} />
              </label>
              <button
                className={`tool-toggle${showBoxes ? " is-active" : ""}`}
                type="button"
                onClick={() => setShowBoxes((value) => !value)}
                aria-pressed={showBoxes}
              >
                <Icon name={showBoxes ? "eye" : "eye-off"} size={15} />
                Boxes
              </button>
            </div>
          </div>

          {!showRedactions && (
            <div className="source-preview-banner">
              <Icon name="warning" size={15} />
              Sensitive content is visible in the review proxy. This is not the immutable original.
            </div>
          )}

          <div className="viewer-wrap">
            <VideoCanvas
              ref={playerRef}
              source={mediaUrl(video.proxy_url) ?? video.proxy_url}
              tracks={tracks}
              overlayTrackIds={overlayTrackIds}
              selectedTrackId={selectedTrackId}
              frameRate={video.fps ?? 30}
              style={previewStyle}
              showRedactions={showRedactions}
              showBoxes={showBoxes}
              manualMode={manualMode}
              draftBox={draftBox}
              onDraftBoxChange={handleDraftBoxChange}
              onManualRegion={(box) => void createManualRegion(box)}
              onBoxCommit={(kind, box) => {
                if (!selectedTrack) return;
                void updateTrack(
                  selectedTrack.track_id,
                  kind === "move" ? "MOVE_REGION" : "RESIZE_REGION",
                  {
                    payload: { bbox: box },
                    frameIndex: currentFrameRef.current,
                    timestampMs: Math.round(currentTimeRef.current),
                    reasonCode: "reviewer_adjustment",
                  },
                );
                setDraftBox(null);
              }}
              onSelectTrack={selectTrack}
              onTimeChange={handleTimeChange}
              onDurationChange={setDurationMs}
              onPlayingChange={handlePlayingChange}
              onMediaError={() => setError("The proxy video could not be decoded by this browser.")}
            />

            <div className="player-controls">
              <button
                className="play-button"
                type="button"
                onClick={() => playerRef.current?.togglePlayback()}
                aria-label={playing ? "Pause video" : "Play video"}
              >
                <Icon name={playing ? "pause" : "play"} size={18} />
              </button>
              <span className="player-time">
                <strong>{formatTime(currentTimeMs)}</strong>
                <i>/</i>
                <span>{formatTime(durationMs)}</span>
              </span>
              <input
                className="player-scrubber"
                type="range"
                min="0"
                max={Math.max(1, durationMs)}
                step={1000 / (video.fps ?? 30)}
                value={Math.min(currentTimeMs, Math.max(1, durationMs))}
                onChange={(event) => playerRef.current?.seekToMs(Number(event.target.value))}
                aria-label="Video position"
                style={{ "--seek": `${(currentTimeMs / Math.max(1, durationMs)) * 100}%` } as React.CSSProperties}
              />
              <span className="frame-readout">Frame {currentFrame}</span>
              <div className="frame-step-controls">
                <button type="button" onClick={() => playerRef.current?.stepFrames(-1)} aria-label="Previous frame">−1</button>
                <button type="button" onClick={() => playerRef.current?.stepFrames(1)} aria-label="Next frame">+1</button>
              </div>
            </div>
          </div>

          <div className={`manual-toolbar${manualMode ? " is-active" : ""}`}>
            <button
              className={`button ${manualMode ? "button-primary" : "button-secondary"}`}
              type="button"
              onClick={toggleManualMode}
            >
              <Icon name={manualMode ? "x" : "plus"} size={16} />
              {manualMode ? "Cancel drawing" : "Add manual region"}
            </button>
            {manualMode ? (
              <label>
                <span>Class</span>
                <select value={manualClass} onChange={(event) => setManualClass(event.target.value)}>
                  <option value="license_plate">License plate</option>
                  <option value="face">Face</option>
                  <option value="scene_text">Sensitive text</option>
                </select>
              </label>
            ) : null}
            {manualMode ? (
              <div className="manual-range-fields">
                <label>
                  <span>Start (s)</span>
                  <input
                    type="number"
                    min="0"
                    max={durationMs / 1000}
                    step="0.1"
                    value={(manualStartMs / 1000).toFixed(1)}
                    onChange={(event) =>
                      setManualStartMs(
                        Math.max(
                          0,
                          Math.min(Number(event.target.value) * 1000, currentTimeMs),
                        ),
                      )
                    }
                  />
                </label>
                <label>
                  <span>End (s)</span>
                  <input
                    type="number"
                    min={currentTimeMs / 1000}
                    max={durationMs / 1000}
                    step="0.1"
                    value={(manualEndMs / 1000).toFixed(1)}
                    onChange={(event) =>
                      setManualEndMs(
                        Math.min(
                          Math.max(1, durationMs),
                          Math.max(Number(event.target.value) * 1000, currentTimeMs),
                        ),
                      )
                    }
                  />
                </label>
              </div>
            ) : (
              <span>Press <kbd>B</kbd> to draw · <kbd>N</kbd> for next unresolved · drag selected boxes to move</span>
            )}
            <div className="edit-history-controls">
              <button type="button" onClick={() => void undo()} disabled={undoStack.length === 0 || mutationBusy} aria-label="Undo">
                <Icon name="undo" size={15} />
              </button>
              <button type="button" onClick={() => void redo()} disabled={redoStack.length === 0 || mutationBusy} aria-label="Redo">
                <Icon name="redo" size={15} />
              </button>
            </div>
          </div>

          <Timeline
            durationMs={durationMs}
            currentTimeMs={currentTimeMs}
            tracks={filteredTracks}
            suggestions={suggestions}
            selectedTrackId={selectedTrackId}
            onSelectTrack={selectTrack}
            onSeek={(time) => playerRef.current?.seekToMs(time)}
          />
        </main>

        <aside className="inspector-panel">
          <div className="inspector-heading">
            <p className="section-kicker">Selection</p>
            <h2>Region details</h2>
          </div>
          {selectedTrack ? (
            <>
              <div className="selected-region-title">
                <span style={{ background: classColours[selectedTrack.class_name] ?? "#8ca9a1" }} />
                <div>
                  <strong>{classLabel(selectedTrack.class_name)}</strong>
                  <small>{selectedTrack.track_id.slice(0, 12)}</small>
                </div>
                <i className={`track-state state-${trackState(selectedTrack).tone}`}>
                  {trackState(selectedTrack).label}
                </i>
              </div>

              <dl className="region-metadata">
                <div><dt>Source</dt><dd>{selectedTrack.source.toLowerCase()}</dd></div>
                <div><dt>Confidence</dt><dd>{selectedTrack.confidence === null ? "Manual" : `${Math.round(selectedTrack.confidence * 100)}%`}</dd></div>
                <div><dt>Duration</dt><dd>{formatTime(selectedTrack.end_ms - selectedTrack.start_ms)}</dd></div>
                <div><dt>Keyframes</dt><dd>{selectedTrack.keyframes.length}</dd></div>
              </dl>

              <div className="track-range-editor">
                <span className="field-label">Track interval</span>
                <div>
                  <label>
                    <span>Start (s)</span>
                    <input
                      type="number"
                      min="0"
                      max={spanEndMs / 1000}
                      step="0.1"
                      value={(spanStartMs / 1000).toFixed(1)}
                      onChange={(event) =>
                        setSpanStartMs(Number(event.target.value) * 1000)
                      }
                    />
                  </label>
                  <label>
                    <span>End (s)</span>
                    <input
                      type="number"
                      min={spanStartMs / 1000}
                      max={durationMs / 1000}
                      step="0.1"
                      value={(spanEndMs / 1000).toFixed(1)}
                      onChange={(event) =>
                        setSpanEndMs(Number(event.target.value) * 1000)
                      }
                    />
                  </label>
                </div>
                <button
                  className="button button-secondary"
                  type="button"
                  disabled={mutationBusy}
                  onClick={() => void applyTrackSpan()}
                >
                  Apply range
                </button>
              </div>

              {(reprocessingStatus || selectedSuggestions.length > 0) && (
                <div className="context-status">
                  <Icon name="sparkles" size={16} />
                  <div className="context-status-body">
                    <strong>Context check</strong>
                    <p>
                      {reprocessingStatus ??
                        `${pendingSelectedSuggestions.length} pending nearby suggestion${
                          pendingSelectedSuggestions.length === 1 ? "" : "s"
                        }. ${
                          selectedTrack.active
                            ? "Automated geometry remains separate until accepted."
                            : "This track is inactive, so its suggestions must be dismissed."
                        }`}
                    </p>
                    {pendingSelectedSuggestions.length > 0 && (
                      <>
                        <div className="context-bulk-actions">
                          {selectedTrack.active && (
                            <button
                              type="button"
                              disabled={mutationBusy}
                              onClick={() => void resolveContextSuggestions("accept")}
                            >
                              Accept all
                            </button>
                          )}
                          <button
                            type="button"
                            disabled={mutationBusy}
                            onClick={() => void resolveContextSuggestions("dismiss")}
                          >
                            Dismiss all
                          </button>
                        </div>
                        <div className="context-suggestion-list">
                          {pendingSelectedSuggestions.map((suggestion) => (
                            <div
                              className="context-suggestion-row"
                              key={suggestion.id}
                            >
                              <button
                                className="context-preview-button"
                                type="button"
                                onClick={() => {
                                  setDraftBox(suggestion.bbox);
                                  playerRef.current?.seekToMs(
                                    suggestion.timestamp_ms,
                                  );
                                }}
                              >
                                <strong>{formatTime(suggestion.timestamp_ms)}</strong>
                                <span>
                                  {Math.round(suggestion.confidence * 100)}% ·{" "}
                                  {suggestion.direction}
                                </span>
                              </button>
                              {selectedTrack.active && (
                                <button
                                  type="button"
                                  aria-label={`Accept context suggestion at ${formatTime(
                                    suggestion.timestamp_ms,
                                  )}`}
                                  disabled={mutationBusy}
                                  onClick={() =>
                                    void resolveContextSuggestions(
                                      "accept",
                                      suggestion.id,
                                    )
                                  }
                                >
                                  <Icon name="check" size={13} />
                                </button>
                              )}
                              <button
                                type="button"
                                aria-label={`Dismiss context suggestion at ${formatTime(
                                  suggestion.timestamp_ms,
                                )}`}
                                disabled={mutationBusy}
                                onClick={() =>
                                  void resolveContextSuggestions(
                                    "dismiss",
                                    suggestion.id,
                                  )
                                }
                              >
                                <Icon name="x" size={13} />
                              </button>
                            </div>
                          ))}
                        </div>
                      </>
                    )}
                  </div>
                </div>
              )}

              {selectedTrack.warning && (
                <div className="continuity-warning">
                  <Icon name="warning" size={16} />
                  <div>
                    <strong>Continuity check</strong>
                    <p>{selectedTrack.warning}</p>
                  </div>
                </div>
              )}

              <div className="inspector-section">
                <label className="field-label" htmlFor="track-class">Region class</label>
                <div className="select-field">
                  <select
                    id="track-class"
                    value={selectedTrack.class_name}
                    disabled={mutationBusy}
                    onChange={(event) =>
                      void updateTrack(selectedTrack.track_id, "CHANGE_CLASS", {
                        payload: { class_name: event.target.value },
                        reasonCode: "reviewer_reclassified",
                      })
                    }
                  >
                    <option value="license_plate">License plate</option>
                    <option value="face">Face</option>
                    <option value="scene_text">Sensitive text</option>
                  </select>
                  <Icon name="chevron-down" size={13} />
                </div>
              </div>

              <div className="decision-section">
                <span className="field-label">Reviewer decision</span>
                {!selectedTrack.accepted ? (
                  <button
                    className="button button-primary decision-primary"
                    type="button"
                    disabled={mutationBusy}
                    onClick={() => void updateTrack(selectedTrack.track_id, "ACCEPT_PROPOSAL")}
                  >
                    <Icon name="check" size={16} />
                    Accept redaction
                  </button>
                ) : (
                  <div className="decision-confirmed">
                    <Icon name="check" size={15} />
                    Decision recorded at revision {snapshot.revision}
                  </div>
                )}
                <div className="decision-grid">
                  <button
                    className={selectedTrack.redacted && selectedTrack.active ? "is-active" : ""}
                    type="button"
                    disabled={mutationBusy}
                    onClick={() => void updateTrack(selectedTrack.track_id, "REDACT_TRACK")}
                  >
                    <Icon name="eye-off" size={16} />
                    Redact
                  </button>
                  <button
                    className={!selectedTrack.redacted && selectedTrack.active ? "is-active" : ""}
                    type="button"
                    disabled={mutationBusy}
                    onClick={() => void updateTrack(selectedTrack.track_id, "RESTORE_TRACK")}
                  >
                    <Icon name="eye" size={16} />
                    Restore
                  </button>
                </div>
                <button
                  className="remove-track-button"
                  type="button"
                  disabled={mutationBusy}
                  onClick={() => void updateTrack(selectedTrack.track_id, "DELETE_FALSE_POSITIVE", { reasonCode: "false_positive" })}
                >
                  <Icon name="trash" size={15} />
                  Mark as false positive
                </button>
              </div>

              <div className="shortcut-card">
                <span>Keyboard</span>
                <div><kbd>A</kbd><p>Accept</p></div>
                <div><kbd>R</kbd><p>Toggle redaction</p></div>
                <div><kbd>B</kbd><p>Draw region</p></div>
                <div><kbd>N</kbd><p>Next unresolved</p></div>
                <div><kbd>J</kbd><kbd>L</kbd><p>Step frame</p></div>
                <div><kbd>Space</kbd><p>Play / pause</p></div>
              </div>
            </>
          ) : (
            <div className="inspector-empty">
              <Icon name="layers" size={23} />
              <strong>Select a region</strong>
              <p>Choose a track from the queue or video to inspect it.</p>
            </div>
          )}

          <div className="evidence-integrity">
            <div><Icon name="shield" size={15} /><strong>Source integrity</strong></div>
            <span title={video.original_sha256 ?? undefined}>{compactHash(video.original_sha256)}</span>
            <small>Original SHA-256</small>
          </div>
        </aside>
      </div>

      <footer className="review-footer">
        <span><i className="privacy-dot" /> {activeRedactions} active redactions</span>
        <span>
          {unresolvedCount === 0 && pendingContextCount === 0
            ? "All proposals reviewed"
            : `${unresolvedCount + pendingContextCount} remaining decisions`}
        </span>
        <span>Revision {snapshot.revision} · source immutable</span>
      </footer>

      <AuditDrawer
        open={auditOpen}
        loading={auditLoading}
        audit={audit}
        onClose={() => setAuditOpen(false)}
        onRefresh={() => void loadAudit()}
      />
      <ExportModal
        key={exportOpen ? "export-open" : "export-closed"}
        open={exportOpen}
        unresolvedCount={unresolvedCount}
        pendingSuggestionCount={pendingContextCount}
        warnings={exportWarnings}
        audioPresent={video.audio_present === true}
        revision={snapshot.revision}
        style={previewStyle}
        exporting={exporting}
        artifact={artifact}
        pollError={exportPollError}
        onStyleChange={setPreviewStyle}
        onExport={() => void startExport()}
        onRetryStatus={retryExportPolling}
        onClose={() => {
          if (!exporting || artifact?.status === "COMPLETED" || artifact?.status === "FAILED") {
            setExportOpen(false);
          }
        }}
      />
    </div>
  );
}
