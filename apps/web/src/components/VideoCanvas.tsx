import {
  forwardRef,
  useCallback,
  useEffect,
  useImperativeHandle,
  useRef,
  useState,
} from "react";

import { boxAtFrame, isUsefulBox, moveBox, normalizeBox, resizeBox } from "../lib/geometry";
import type { NormalizedBox, RedactionStyle, ReviewTrack } from "../types";

export interface VideoCanvasHandle {
  togglePlayback: () => void;
  play: () => void;
  pause: () => void;
  seekToMs: (timeMs: number) => void;
  stepFrames: (frames: number) => void;
}

interface VideoCanvasProps {
  source: string;
  tracks: ReviewTrack[];
  overlayTrackIds: Set<string>;
  selectedTrackId: string | null;
  frameRate: number;
  style: RedactionStyle;
  showRedactions: boolean;
  showBoxes: boolean;
  manualMode: boolean;
  draftBox: NormalizedBox | null;
  onDraftBoxChange: (box: NormalizedBox | null) => void;
  onManualRegion: (box: NormalizedBox) => void;
  onBoxCommit: (kind: "move" | "resize", box: NormalizedBox) => void;
  onSelectTrack: (trackId: string) => void;
  onTimeChange: (timeMs: number, frame: number) => void;
  onDurationChange: (durationMs: number) => void;
  onPlayingChange: (playing: boolean) => void;
  onMediaError: () => void;
}

interface PointerEdit {
  pointerId: number;
  kind: "move" | "resize";
  start: { x: number; y: number };
  original: NormalizedBox;
}

interface SurfaceSize {
  width: number;
  height: number;
}

function boxStyle(box: NormalizedBox) {
  return {
    left: `${box.x1 * 100}%`,
    top: `${box.y1 * 100}%`,
    width: `${(box.x2 - box.x1) * 100}%`,
    height: `${(box.y2 - box.y1) * 100}%`,
  };
}

function paddedBox(box: NormalizedBox, className: string): NormalizedBox {
  const width = box.x2 - box.x1;
  const height = box.y2 - box.y1;
  const padding =
    className === "license_plate" ? 0.15 : className === "face" ? 0.25 : 0.12;
  return {
    x1: Math.max(0, box.x1 - width * padding),
    y1: Math.max(0, box.y1 - height * padding),
    x2: Math.min(1, box.x2 + width * padding),
    y2: Math.min(1, box.y2 + height * padding),
  };
}

function redactRegion(
  context: CanvasRenderingContext2D,
  source: HTMLVideoElement,
  box: NormalizedBox,
  style: RedactionStyle,
) {
  const canvas = context.canvas;
  const x = Math.max(0, Math.min(canvas.width - 1, Math.floor(box.x1 * canvas.width)));
  const y = Math.max(0, Math.min(canvas.height - 1, Math.floor(box.y1 * canvas.height)));
  const x2 = Math.max(x + 1, Math.min(canvas.width, Math.ceil(box.x2 * canvas.width)));
  const y2 = Math.max(y + 1, Math.min(canvas.height, Math.ceil(box.y2 * canvas.height)));
  const width = x2 - x;
  const height = y2 - y;

  if (style === "black_box") {
    context.fillStyle = "#000";
    context.fillRect(x, y, width, height);
    return;
  }

  if (style === "pixelate") {
    const sample = document.createElement("canvas");
    const sampleContext = sample.getContext("2d");
    if (!sampleContext) return;
    sample.width = Math.max(1, Math.floor(width / 12));
    sample.height = Math.max(1, Math.floor(height / 12));
    sampleContext.drawImage(source, x, y, width, height, 0, 0, sample.width, sample.height);
    context.save();
    context.imageSmoothingEnabled = false;
    context.drawImage(sample, 0, 0, sample.width, sample.height, x, y, width, height);
    context.restore();
    return;
  }

  let kernel = Math.max(3, Math.floor(Math.min(width, height) / 3));
  if (kernel % 2 === 0) kernel += 1;
  const sigma = Math.max(1, 0.3 * ((kernel - 1) * 0.5 - 1) + 0.8);
  context.save();
  context.beginPath();
  context.rect(x, y, width, height);
  context.clip();
  context.filter = `blur(${sigma}px)`;
  context.drawImage(source, 0, 0, canvas.width, canvas.height);
  context.restore();
}

export const VideoCanvas = forwardRef<VideoCanvasHandle, VideoCanvasProps>(
  function VideoCanvas(
    {
      source,
      tracks,
      overlayTrackIds,
      selectedTrackId,
      frameRate,
      style,
      showRedactions,
      showBoxes,
      manualMode,
      draftBox,
      onDraftBoxChange,
      onManualRegion,
      onBoxCommit,
      onSelectTrack,
      onTimeChange,
      onDurationChange,
      onPlayingChange,
      onMediaError,
    },
    forwardedRef,
  ) {
    const videoRef = useRef<HTMLVideoElement>(null);
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const stageRef = useRef<HTMLDivElement>(null);
    const animationRef = useRef<number | null>(null);
    const lastFrameRef = useRef(-1);
    const manualStartRef = useRef<{ x: number; y: number; pointerId: number } | null>(null);
    const pointerEditRef = useRef<PointerEdit | null>(null);
    const draftBoxRef = useRef<NormalizedBox | null>(draftBox);
    const [frame, setFrame] = useState(0);
    const [ready, setReady] = useState(false);
    const [mediaSize, setMediaSize] = useState<SurfaceSize | null>(null);
    const [surfaceSize, setSurfaceSize] = useState<SurfaceSize | null>(null);

    useEffect(() => {
      draftBoxRef.current = draftBox;
    }, [draftBox]);

    const updateDraftBox = useCallback(
      (box: NormalizedBox | null) => {
        draftBoxRef.current = box;
        onDraftBoxChange(box);
      },
      [onDraftBoxChange],
    );

    const fitSurface = useCallback(() => {
      const stage = stageRef.current;
      const container = stage?.parentElement;
      if (!stage || !container || !mediaSize) return;

      const rootFontSize =
        Number.parseFloat(window.getComputedStyle(document.documentElement).fontSize) || 16;
      const containerStyle = window.getComputedStyle(container);
      const horizontalPadding =
        Number.parseFloat(containerStyle.paddingLeft) +
        Number.parseFloat(containerStyle.paddingRight);
      const verticalPadding =
        Number.parseFloat(containerStyle.paddingTop) +
        Number.parseFloat(containerStyle.paddingBottom);
      const controls = container.querySelector<HTMLElement>(".player-controls");
      const controlsHeight = controls?.getBoundingClientRect().height ?? 0;
      const availableWidth = Math.max(1, container.clientWidth - horizontalPadding);
      const containerHeight = Math.max(
        1,
        container.clientHeight - verticalPadding - controlsHeight,
      );
      const compact = window.innerWidth <= 800;
      const viewportHeight = Math.max(
        rootFontSize * 10,
        compact ? window.innerHeight * 0.45 : window.innerHeight - rootFontSize * 19.5,
      );
      const availableHeight = Math.max(1, Math.min(containerHeight, viewportHeight));
      const maxWidth = Math.min(availableWidth, rootFontSize * 66);
      const scale = Math.min(maxWidth / mediaSize.width, availableHeight / mediaSize.height);
      const next = {
        width: Math.max(1, Math.round(mediaSize.width * scale)),
        height: Math.max(1, Math.round(mediaSize.height * scale)),
      };
      setSurfaceSize((current) =>
        current?.width === next.width && current.height === next.height ? current : next,
      );
    }, [mediaSize]);

    useEffect(() => {
      if (!mediaSize) return;
      const stage = stageRef.current;
      const container = stage?.parentElement;
      if (!stage || !container) return;
      fitSurface();
      const observer =
        typeof ResizeObserver === "undefined" ? null : new ResizeObserver(fitSurface);
      observer?.observe(container);
      window.addEventListener("resize", fitSurface);
      return () => {
        observer?.disconnect();
        window.removeEventListener("resize", fitSurface);
      };
    }, [fitSurface, mediaSize]);

    const draw = useCallback(() => {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      if (!video || !canvas || video.readyState < 2) return;
      if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
      }
      const context = canvas.getContext("2d");
      if (!context) return;
      context.drawImage(video, 0, 0, canvas.width, canvas.height);
      const currentFrame = Math.max(0, Math.round(video.currentTime * frameRate));
      if (showRedactions) {
        tracks.forEach((track) => {
          if (!track.active || !track.redacted) return;
          const box = boxAtFrame(track, currentFrame);
          if (box) redactRegion(context, video, paddedBox(box, track.class_name), style);
        });
      }
      if (currentFrame !== lastFrameRef.current) {
        lastFrameRef.current = currentFrame;
        setFrame(currentFrame);
        onTimeChange(video.currentTime * 1000, currentFrame);
      }
    }, [frameRate, onTimeChange, showRedactions, style, tracks]);

    const animate = useCallback(() => {
      draw();
      if (videoRef.current && !videoRef.current.paused) {
        animationRef.current = requestAnimationFrame(animate);
      }
    }, [draw]);

    useEffect(() => {
      draw();
    }, [draw, draftBox]);

    useEffect(
      () => () => {
        if (animationRef.current !== null) cancelAnimationFrame(animationRef.current);
      },
      [],
    );

    useImperativeHandle(
      forwardedRef,
      () => ({
        togglePlayback() {
          const video = videoRef.current;
          if (!video) return;
          if (video.paused) void video.play();
          else video.pause();
        },
        play() {
          if (videoRef.current) void videoRef.current.play();
        },
        pause() {
          videoRef.current?.pause();
        },
        seekToMs(timeMs: number) {
          const video = videoRef.current;
          if (!video) return;
          video.currentTime = Math.max(0, Math.min(video.duration || 0, timeMs / 1000));
          draw();
        },
        stepFrames(frames: number) {
          const video = videoRef.current;
          if (!video) return;
          video.pause();
          video.currentTime = Math.max(
            0,
            Math.min(video.duration || 0, video.currentTime + frames / frameRate),
          );
          draw();
        },
      }),
      [draw, frameRate],
    );

    const pointFromEvent = (event: React.PointerEvent): { x: number; y: number } => {
      const bounds = canvasRef.current?.getBoundingClientRect();
      if (!bounds) return { x: 0, y: 0 };
      return {
        x: Math.min(1, Math.max(0, (event.clientX - bounds.left) / bounds.width)),
        y: Math.min(1, Math.max(0, (event.clientY - bounds.top) / bounds.height)),
      };
    };

    const selectedTrack = tracks.find((track) => track.track_id === selectedTrackId) ?? null;
    const selectedBox =
      draftBox ?? (selectedTrack ? boxAtFrame(selectedTrack, frame) : null);

    const visibleBoxes = tracks
      .filter((track) => overlayTrackIds.has(track.track_id))
      .map((track) => ({ track, box: boxAtFrame(track, frame) }))
      .filter(
        (item): item is { track: ReviewTrack; box: NormalizedBox } => item.box !== null,
      );

    const beginEdit = (
      event: React.PointerEvent,
      kind: "move" | "resize",
      box: NormalizedBox,
    ) => {
      if (manualMode) return;
      event.preventDefault();
      event.stopPropagation();
      event.currentTarget.setPointerCapture(event.pointerId);
      pointerEditRef.current = {
        pointerId: event.pointerId,
        kind,
        start: pointFromEvent(event),
        original: box,
      };
      updateDraftBox(box);
    };

    const updateEdit = (event: React.PointerEvent) => {
      const edit = pointerEditRef.current;
      if (!edit || edit.pointerId !== event.pointerId) return;
      const point = pointFromEvent(event);
      const dx = point.x - edit.start.x;
      const dy = point.y - edit.start.y;
      updateDraftBox(
        edit.kind === "move"
          ? moveBox(edit.original, dx, dy)
          : resizeBox(edit.original, dx, dy),
      );
    };

    const finishEdit = (event: React.PointerEvent) => {
      const edit = pointerEditRef.current;
      if (!edit || edit.pointerId !== event.pointerId) return;
      pointerEditRef.current = null;
      const committed = draftBoxRef.current;
      if (committed) onBoxCommit(edit.kind, committed);
    };

    const cancelEdit = (event: React.PointerEvent) => {
      const edit = pointerEditRef.current;
      if (!edit || edit.pointerId !== event.pointerId) return;
      pointerEditRef.current = null;
      updateDraftBox(null);
    };

    const resizeFromKeyboard = (
      event: React.KeyboardEvent<HTMLButtonElement>,
      box: NormalizedBox,
    ) => {
      if (event.key === " " || event.key === "Enter") {
        event.stopPropagation();
        return;
      }
      const directions: Record<string, { dx: number; dy: number }> = {
        ArrowLeft: { dx: -1, dy: 0 },
        ArrowRight: { dx: 1, dy: 0 },
        ArrowUp: { dx: 0, dy: -1 },
        ArrowDown: { dx: 0, dy: 1 },
      };
      const direction = directions[event.key];
      if (!direction) return;
      event.preventDefault();
      event.stopPropagation();
      const amount = event.shiftKey ? 0.02 : 0.005;
      const resized = resizeBox(
        draftBoxRef.current ?? box,
        direction.dx * amount,
        direction.dy * amount,
      );
      updateDraftBox(resized);
    };

    const finishKeyboardResize = (event: React.KeyboardEvent<HTMLButtonElement>) => {
      if (!event.key.startsWith("Arrow")) return;
      event.preventDefault();
      event.stopPropagation();
      const resized = draftBoxRef.current;
      if (resized) onBoxCommit("resize", resized);
    };

    return (
      <div
        className={`video-stage${manualMode ? " is-drawing" : ""}`}
        ref={stageRef}
        style={
          surfaceSize
            ? { width: `${surfaceSize.width}px`, height: `${surfaceSize.height}px` }
            : undefined
        }
      >
        <video
          ref={videoRef}
          className="video-source"
          src={source}
          playsInline
          preload="auto"
          onLoadedMetadata={(event) => {
            const video = event.currentTarget;
            setMediaSize({ width: video.videoWidth, height: video.videoHeight });
            onDurationChange(video.duration * 1000);
          }}
          onLoadedData={() => {
            setReady(true);
            draw();
          }}
          onSeeked={draw}
          onPlay={() => {
            onPlayingChange(true);
            animationRef.current = requestAnimationFrame(animate);
          }}
          onPause={() => {
            onPlayingChange(false);
            draw();
          }}
          onEnded={() => onPlayingChange(false)}
          onError={onMediaError}
        />
        <canvas ref={canvasRef} className="video-canvas" aria-label="Video review preview" />

        {!ready && (
          <div className="video-loading" role="status">
            <span className="spinner" />
            <span>Loading private review copy</span>
          </div>
        )}

        {showBoxes && (
          <div className="region-layer">
            {visibleBoxes.map(({ track, box }) => {
              const selected = track.track_id === selectedTrackId;
              return (
                <div
                  className={`region-outline ${selected ? "is-selected" : ""} ${
                    track.accepted ? "is-reviewed" : "needs-review"
                  }`}
                  style={boxStyle(selected && draftBox ? draftBox : box)}
                  key={track.track_id}
                >
                  <button
                    className="region-select-button"
                    type="button"
                    onClick={(event) => {
                      event.stopPropagation();
                      onSelectTrack(track.track_id);
                    }}
                    onPointerDown={(event) => {
                      onSelectTrack(track.track_id);
                      if (selected) beginEdit(event, "move", draftBox ?? box);
                    }}
                    onPointerMove={updateEdit}
                    onPointerUp={finishEdit}
                    onPointerCancel={cancelEdit}
                    onKeyDown={(event) => {
                      if (event.key === " ") event.stopPropagation();
                    }}
                    aria-label={`Select ${track.class_name.replaceAll("_", " ")} track`}
                  >
                    <span className="region-label">
                      {track.class_name.replaceAll("_", " ")}
                      {track.confidence !== null &&
                        ` · ${Math.round(track.confidence * 100)}%`}
                    </span>
                  </button>
                  {selected && (
                    <button
                      className="resize-handle"
                      type="button"
                      aria-label="Resize region"
                      onPointerDown={(event) => beginEdit(event, "resize", draftBox ?? box)}
                      onPointerMove={updateEdit}
                      onPointerUp={finishEdit}
                      onPointerCancel={cancelEdit}
                      onKeyDown={(event) => resizeFromKeyboard(event, draftBox ?? box)}
                      onKeyUp={finishKeyboardResize}
                    />
                  )}
                </div>
              );
            })}
          </div>
        )}

        {manualMode && (
          <div
            className="manual-draw-layer"
            role="application"
            aria-label="Draw a redaction region"
            onPointerDown={(event) => {
              const start = pointFromEvent(event);
              event.currentTarget.setPointerCapture(event.pointerId);
              manualStartRef.current = { ...start, pointerId: event.pointerId };
              updateDraftBox({ x1: start.x, y1: start.y, x2: start.x, y2: start.y });
            }}
            onPointerMove={(event) => {
              const start = manualStartRef.current;
              if (!start || start.pointerId !== event.pointerId) return;
              updateDraftBox(normalizeBox(start, pointFromEvent(event)));
            }}
            onPointerUp={(event) => {
              const start = manualStartRef.current;
              if (!start || start.pointerId !== event.pointerId) return;
              const box = normalizeBox(start, pointFromEvent(event));
              manualStartRef.current = null;
              if (isUsefulBox(box)) onManualRegion(box);
              else updateDraftBox(null);
            }}
            onPointerCancel={(event) => {
              const start = manualStartRef.current;
              if (!start || start.pointerId !== event.pointerId) return;
              manualStartRef.current = null;
              updateDraftBox(null);
            }}
          >
            {draftBox && <span className="manual-region-preview" style={boxStyle(draftBox)} />}
            <span className="draw-instruction">Drag around the sensitive region</span>
          </div>
        )}

        {!manualMode && selectedBox && !showBoxes && (
          <span className="selected-region-fallback" style={boxStyle(selectedBox)} />
        )}
      </div>
    );
  },
);
