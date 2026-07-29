import { useMemo, useRef, useState } from "react";

import { formatTime } from "../lib/geometry";
import { boundedTrackWindow, sampleKeyframes } from "../lib/reviewPerformance";
import type { ReprocessingSuggestion, ReviewTrack } from "../types";
import { Icon } from "./Icon";

interface TimelineProps {
  durationMs: number;
  currentTimeMs: number;
  tracks: ReviewTrack[];
  suggestions: ReprocessingSuggestion[];
  selectedTrackId: string | null;
  onSelectTrack: (trackId: string) => void;
  onSeek: (timeMs: number) => void;
}

const classColours: Record<string, string> = {
  license_plate: "#61d6b0",
  face: "#af98ff",
  scene_text: "#f4b85d",
};

const maximumTimelineTracks = 60;
const maximumKeyframeMarkers = 32;

function shortClassName(name: string): string {
  if (name === "license_plate") return "Plate";
  return name.charAt(0).toUpperCase() + name.slice(1).replaceAll("_", " ");
}

export function Timeline({
  durationMs,
  currentTimeMs,
  tracks,
  suggestions,
  selectedTrackId,
  onSelectTrack,
  onSeek,
}: TimelineProps) {
  const timelineRef = useRef<HTMLDivElement>(null);
  const [zoom, setZoom] = useState(1);
  const totalDuration = Math.max(1, durationMs);
  const widthPercent = Math.max(100, zoom * 100);
  const ticks = useMemo(() => {
    const count = Math.max(5, Math.min(12, Math.ceil((durationMs / 1000) * zoom / 4)));
    return Array.from({ length: count + 1 }, (_, index) => ({
      left: (index / count) * 100,
      time: (durationMs * index) / count,
    }));
  }, [durationMs, zoom]);
  const pendingSuggestions = useMemo(
    () => suggestions.filter((item) => item.status === "PENDING"),
    [suggestions],
  );
  const displayTracks = useMemo(
    () => boundedTrackWindow(tracks, selectedTrackId, maximumTimelineTracks),
    [selectedTrackId, tracks],
  );
  const suggestionsByTrack = useMemo(() => {
    const grouped = new Map<string, ReprocessingSuggestion[]>();
    pendingSuggestions.forEach((suggestion) => {
      const items = grouped.get(suggestion.track_id);
      if (items) items.push(suggestion);
      else grouped.set(suggestion.track_id, [suggestion]);
    });
    return grouped;
  }, [pendingSuggestions]);
  const keyframesByTrack = useMemo(
    () =>
      new Map(
        displayTracks.map((track) => [
          track.track_id,
          sampleKeyframes(track.keyframes, maximumKeyframeMarkers),
        ]),
      ),
    [displayTracks],
  );

  const seekFromPointer = (clientX: number) => {
    const element = timelineRef.current;
    if (!element) return;
    const bounds = element.getBoundingClientRect();
    const relativeX = clientX - bounds.left + element.scrollLeft;
    onSeek((Math.min(1, Math.max(0, relativeX / element.scrollWidth)) || 0) * totalDuration);
  };

  return (
    <section className="timeline-panel" aria-label="Track timeline">
      <div className="timeline-toolbar">
        <div>
          <span className="timeline-title">Timeline</span>
          <span className="timeline-count">
            {displayTracks.length === tracks.length
              ? `${tracks.length} visible tracks`
              : `${displayTracks.length} of ${tracks.length} visible tracks`}
          </span>
        </div>
        <label className="zoom-control">
          <Icon name="zoom-in" size={15} />
          <span className="visually-hidden">Timeline zoom</span>
          <input
            type="range"
            min="1"
            max="4"
            step="0.25"
            value={zoom}
            onChange={(event) => setZoom(Number(event.target.value))}
          />
        </label>
      </div>

      <div className="timeline-shell">
        <div className="timeline-labels">
          <span className="ruler-spacer">Track</span>
          {displayTracks.map((track) => (
            <button
              type="button"
              className={track.track_id === selectedTrackId ? "is-selected" : ""}
              onClick={() => onSelectTrack(track.track_id)}
              onKeyDown={(event) => {
                if (event.key === " ") event.stopPropagation();
              }}
              key={track.track_id}
              title={track.class_name.replaceAll("_", " ")}
            >
              <i style={{ background: classColours[track.class_name] ?? "#8ca9a1" }} />
              <span>{shortClassName(track.class_name)}</span>
            </button>
          ))}
        </div>

        <div className="timeline-scroll" ref={timelineRef}>
          <div
            className="timeline-content"
            style={{ width: `${widthPercent}%` }}
            onPointerDown={(event) => {
              if (
                (event.target as HTMLElement).closest(
                  ".track-span, .context-timeline-marker",
                )
              ) {
                return;
              }
              event.currentTarget.setPointerCapture(event.pointerId);
              seekFromPointer(event.clientX);
            }}
            onPointerMove={(event) => {
              if (event.currentTarget.hasPointerCapture(event.pointerId)) {
                seekFromPointer(event.clientX);
              }
            }}
          >
            <div className="timeline-ruler">
              {ticks.map((tick) => (
                <span style={{ left: `${tick.left}%` }} key={tick.left}>
                  <i />
                  {formatTime(tick.time).slice(0, 5)}
                </span>
              ))}
            </div>
            {displayTracks.map((track) => {
              const left = (track.start_ms / totalDuration) * 100;
              const width = Math.max(0.5, ((track.end_ms - track.start_ms) / totalDuration) * 100);
              const color = classColours[track.class_name] ?? "#8ca9a1";
              return (
                <div className="timeline-lane" key={track.track_id}>
                  <button
                    type="button"
                    className={`track-span${track.track_id === selectedTrackId ? " is-selected" : ""}${
                      !track.active ? " is-inactive" : ""
                    }`}
                    style={{ left: `${left}%`, width: `${width}%`, "--track-color": color } as React.CSSProperties}
                    onClick={(event) => {
                      event.stopPropagation();
                      onSelectTrack(track.track_id);
                      onSeek(Math.max(track.start_ms, Math.min(track.end_ms, currentTimeMs)));
                    }}
                    onKeyDown={(event) => {
                      if (event.key === " ") event.stopPropagation();
                    }}
                    title={`${shortClassName(track.class_name)} · ${formatTime(track.start_ms)}–${formatTime(track.end_ms)}`}
                  >
                    {(keyframesByTrack.get(track.track_id) ?? []).map((keyframe) => (
                      <i
                        className="keyframe-dot"
                        style={{
                          left: `${((keyframe.timestamp_ms - track.start_ms) / Math.max(1, track.end_ms - track.start_ms)) * 100}%`,
                        }}
                        key={keyframe.frame_index}
                      />
                    ))}
                  </button>
                  {(suggestionsByTrack.get(track.track_id) ?? [])
                    .map((suggestion) => (
                      <button
                        className="context-timeline-marker"
                        type="button"
                        style={{
                          left: `${(suggestion.timestamp_ms / totalDuration) * 100}%`,
                        }}
                        aria-label={`Pending context suggestion at ${formatTime(
                          suggestion.timestamp_ms,
                        )}`}
                        title={`Context suggestion · ${Math.round(
                          suggestion.confidence * 100,
                        )}% confidence`}
                        onClick={(event) => {
                          event.stopPropagation();
                          onSelectTrack(track.track_id);
                          onSeek(suggestion.timestamp_ms);
                        }}
                        key={suggestion.id}
                      >
                        <span className="visually-hidden">Context suggestion</span>
                      </button>
                    ))}
                </div>
              );
            })}
            <span
              className="timeline-playhead"
              style={{ left: `${(currentTimeMs / totalDuration) * 100}%` }}
            >
              <i />
            </span>
          </div>
        </div>
      </div>
    </section>
  );
}
