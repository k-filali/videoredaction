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

interface ClassSegment {
  startMs: number;
  endMs: number;
  count: number;
  unaccepted: number;
  representativeId: string;
}

const classColours: Record<string, string> = {
  license_plate: "#61d6b0",
  face: "#af98ff",
  scene_text: "#f4b85d",
};

const maximumDetailTracks = 40;
const maximumKeyframeMarkers = 32;
const maximumExceptionMarkers = 120;

function shortClassName(name: string): string {
  if (name === "license_plate") return "Plate";
  return name.charAt(0).toUpperCase() + name.slice(1).replaceAll("_", " ");
}

function buildSegments(tracks: ReviewTrack[]): ClassSegment[] {
  const ordered = [...tracks].sort((a, b) => a.start_ms - b.start_ms);
  const segments: ClassSegment[] = [];
  for (const track of ordered) {
    const current = segments[segments.length - 1];
    if (current && track.start_ms <= current.endMs) {
      current.endMs = Math.max(current.endMs, track.end_ms);
      current.count += 1;
      if (!track.accepted) {
        if (current.unaccepted === 0) current.representativeId = track.track_id;
        current.unaccepted += 1;
      }
    } else {
      segments.push({
        startMs: track.start_ms,
        endMs: track.end_ms,
        count: 1,
        unaccepted: track.accepted ? 0 : 1,
        representativeId: track.track_id,
      });
    }
  }
  return segments;
}

function sampleEvenly<T>(items: T[], maximum: number): T[] {
  if (items.length <= maximum) return items;
  const step = items.length / maximum;
  return Array.from(
    { length: maximum },
    (_, index) => items[Math.floor(index * step)]!,
  );
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
  const [expandedClass, setExpandedClass] = useState<string | null>(null);
  const totalDuration = Math.max(1, durationMs);
  const widthPercent = Math.max(100, zoom * 100);

  const ticks = useMemo(() => {
    const count = Math.max(5, Math.min(12, Math.ceil((durationMs / 1000) * zoom / 4)));
    return Array.from({ length: count + 1 }, (_, index) => ({
      left: (index / count) * 100,
      time: (durationMs * index) / count,
    }));
  }, [durationMs, zoom]);

  const tracksByClass = useMemo(() => {
    const grouped = new Map<string, ReviewTrack[]>();
    tracks.forEach((track) => {
      const items = grouped.get(track.class_name);
      if (items) items.push(track);
      else grouped.set(track.class_name, [track]);
    });
    return new Map(
      [...grouped.entries()].sort(([a], [b]) => a.localeCompare(b)),
    );
  }, [tracks]);

  const segmentsByClass = useMemo(
    () =>
      new Map(
        [...tracksByClass.entries()].map(([className, classTracks]) => [
          className,
          buildSegments(classTracks),
        ]),
      ),
    [tracksByClass],
  );

  const selectedTrack = useMemo(
    () => tracks.find((track) => track.track_id === selectedTrackId) ?? null,
    [selectedTrackId, tracks],
  );

  const [lastSelectedId, setLastSelectedId] = useState<string | null>(null);
  if (selectedTrackId !== lastSelectedId) {
    setLastSelectedId(selectedTrackId);
    if (selectedTrack) setExpandedClass(selectedTrack.class_name);
  }

  const detailTracks = useMemo(() => {
    if (!expandedClass) return [];
    const classTracks = tracksByClass.get(expandedClass) ?? [];
    return boundedTrackWindow(classTracks, selectedTrackId, maximumDetailTracks);
  }, [expandedClass, selectedTrackId, tracksByClass]);

  const detailKeyframes = useMemo(
    () =>
      new Map(
        detailTracks.map((track) => [
          track.track_id,
          sampleKeyframes(track.keyframes, maximumKeyframeMarkers),
        ]),
      ),
    [detailTracks],
  );

  const pendingSuggestions = useMemo(
    () => suggestions.filter((item) => item.status === "PENDING"),
    [suggestions],
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

  const exceptions = useMemo(() => {
    const markers = tracks
      .filter((track) => !track.accepted)
      .map((track) => ({
        key: `track-${track.track_id}`,
        timeMs: track.start_ms,
        trackId: track.track_id,
        className: track.class_name,
        label: `Unconfirmed ${shortClassName(track.class_name).toLowerCase()} at ${formatTime(track.start_ms)}`,
      }))
      .concat(
        pendingSuggestions.map((suggestion) => ({
          key: `suggestion-${suggestion.id}`,
          timeMs: suggestion.timestamp_ms,
          trackId: suggestion.track_id,
          className: suggestion.class_name,
          label: `Pending context suggestion at ${formatTime(suggestion.timestamp_ms)}`,
        })),
      )
      .sort((a, b) => a.timeMs - b.timeMs);
    return sampleEvenly(markers, maximumExceptionMarkers);
  }, [pendingSuggestions, tracks]);

  const seekFromPointer = (clientX: number) => {
    const element = timelineRef.current;
    if (!element) return;
    const bounds = element.getBoundingClientRect();
    const relativeX = clientX - bounds.left + element.scrollLeft;
    onSeek((Math.min(1, Math.max(0, relativeX / element.scrollWidth)) || 0) * totalDuration);
  };

  interface LaneRow {
    key: string;
    kind: "class" | "detail";
    className: string;
    track?: ReviewTrack;
  }

  const rows = useMemo(() => {
    const built: LaneRow[] = [];
    for (const className of tracksByClass.keys()) {
      built.push({ key: `class-${className}`, kind: "class", className });
      if (className === expandedClass) {
        detailTracks.forEach((track) =>
          built.push({
            key: `detail-${track.track_id}`,
            kind: "detail",
            className,
            track,
          }),
        );
      }
    }
    return built;
  }, [detailTracks, expandedClass, tracksByClass]);

  const totalUnaccepted = useMemo(
    () => tracks.filter((track) => !track.accepted).length,
    [tracks],
  );

  return (
    <section className="timeline-panel" aria-label="Redaction timeline">
      <div className="timeline-toolbar">
        <div>
          <span className="timeline-title">Timeline</span>
          <span className="timeline-count">
            {tracks.length} tracks in {tracksByClass.size}{" "}
            {tracksByClass.size === 1 ? "lane" : "lanes"}
            {totalUnaccepted > 0 ? ` · ${totalUnaccepted} unconfirmed` : " · all confirmed"}
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
          <span className="exception-strip-label">
            <Icon name="warning" size={12} />
            <span>Review</span>
          </span>
          {rows.map((row) =>
            row.kind === "class" ? (
              <button
                type="button"
                className={`class-lane-label${row.className === expandedClass ? " is-expanded" : ""}`}
                onClick={() =>
                  setExpandedClass((current) =>
                    current === row.className ? null : row.className,
                  )
                }
                onKeyDown={(event) => {
                  if (event.key === " ") event.stopPropagation();
                }}
                key={row.key}
                title={`${shortClassName(row.className)} · ${
                  (tracksByClass.get(row.className) ?? []).length
                } tracks`}
              >
                <i style={{ background: classColours[row.className] ?? "#8ca9a1" }} />
                <span>{shortClassName(row.className)}</span>
                <em>{(tracksByClass.get(row.className) ?? []).length}</em>
                <Icon
                  name="chevron-down"
                  size={12}
                />
              </button>
            ) : (
              <button
                type="button"
                className={`detail-lane-label${
                  row.track!.track_id === selectedTrackId ? " is-selected" : ""
                }`}
                onClick={() => onSelectTrack(row.track!.track_id)}
                onKeyDown={(event) => {
                  if (event.key === " ") event.stopPropagation();
                }}
                key={row.key}
                title={`${shortClassName(row.className)} · ${formatTime(row.track!.start_ms)}–${formatTime(row.track!.end_ms)}`}
              >
                <i
                  className={row.track!.accepted ? "is-accepted" : "is-pending"}
                  style={{ background: classColours[row.className] ?? "#8ca9a1" }}
                />
                <span>{formatTime(row.track!.start_ms).slice(0, 5)}</span>
              </button>
            ),
          )}
        </div>

        <div className="timeline-scroll" ref={timelineRef}>
          <div
            className="timeline-content"
            style={{ width: `${widthPercent}%` }}
            onPointerDown={(event) => {
              if (
                (event.target as HTMLElement).closest(
                  ".track-span, .class-segment, .context-timeline-marker, .exception-marker",
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

            <div className="timeline-lane exception-strip">
              {exceptions.map((marker) => (
                <button
                  className="exception-marker"
                  type="button"
                  style={{
                    left: `${(marker.timeMs / totalDuration) * 100}%`,
                    "--marker-color":
                      classColours[marker.className] ?? "#8ca9a1",
                  } as React.CSSProperties}
                  aria-label={marker.label}
                  title={marker.label}
                  onClick={(event) => {
                    event.stopPropagation();
                    onSelectTrack(marker.trackId);
                    onSeek(marker.timeMs);
                  }}
                  key={marker.key}
                >
                  <span className="visually-hidden">{marker.label}</span>
                </button>
              ))}
            </div>

            {rows.map((row) => {
              if (row.kind === "class") {
                const segments = segmentsByClass.get(row.className) ?? [];
                const color = classColours[row.className] ?? "#8ca9a1";
                return (
                  <div className="timeline-lane" key={row.key}>
                    {segments.map((segment) => {
                      const left = (segment.startMs / totalDuration) * 100;
                      const width = Math.max(
                        0.5,
                        ((segment.endMs - segment.startMs) / totalDuration) * 100,
                      );
                      return (
                        <button
                          type="button"
                          className={`class-segment${segment.unaccepted > 0 ? " has-pending" : ""}`}
                          style={{
                            left: `${left}%`,
                            width: `${width}%`,
                            "--track-color": color,
                          } as React.CSSProperties}
                          onClick={(event) => {
                            event.stopPropagation();
                            setExpandedClass(row.className);
                            onSelectTrack(segment.representativeId);
                            onSeek(
                              Math.max(
                                segment.startMs,
                                Math.min(segment.endMs, currentTimeMs),
                              ),
                            );
                          }}
                          onKeyDown={(event) => {
                            if (event.key === " ") event.stopPropagation();
                          }}
                          title={`${segment.count} ${shortClassName(row.className).toLowerCase()} ${
                            segment.count === 1 ? "track" : "tracks"
                          } · ${formatTime(segment.startMs)}–${formatTime(segment.endMs)}${
                            segment.unaccepted > 0
                              ? ` · ${segment.unaccepted} unconfirmed`
                              : ""
                          }`}
                          key={`${row.key}-${segment.startMs}`}
                        >
                          {segment.count > 1 && (
                            <span className="segment-count">×{segment.count}</span>
                          )}
                        </button>
                      );
                    })}
                  </div>
                );
              }

              const track = row.track!;
              const left = (track.start_ms / totalDuration) * 100;
              const width = Math.max(
                0.5,
                ((track.end_ms - track.start_ms) / totalDuration) * 100,
              );
              const color = classColours[track.class_name] ?? "#8ca9a1";
              return (
                <div className="timeline-lane is-detail" key={row.key}>
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
                    {(detailKeyframes.get(track.track_id) ?? []).map((keyframe) => (
                      <i
                        className="keyframe-dot"
                        style={{
                          left: `${((keyframe.timestamp_ms - track.start_ms) / Math.max(1, track.end_ms - track.start_ms)) * 100}%`,
                        }}
                        key={keyframe.frame_index}
                      />
                    ))}
                  </button>
                  {(suggestionsByTrack.get(track.track_id) ?? []).map((suggestion) => (
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
