import { describe, expect, it } from "vitest";

import type { ReviewKeyframe, ReviewTrack } from "../types";
import {
  boundedTrackWindow,
  buildTrackFrameIndex,
  sampleKeyframes,
  tracksAtFrame,
} from "./reviewPerformance";

function track(id: number, start: number, end: number): ReviewTrack {
  return {
    track_id: `track-${id}`,
    class_name: "face",
    source: "MODEL",
    active: true,
    redacted: true,
    accepted: false,
    start_frame: start,
    end_frame: end,
    start_ms: start * 40,
    end_ms: end * 40,
    confidence: 0.9,
    warning: null,
    keyframes: [],
  };
}

describe("review performance helpers", () => {
  it("queries only tracks that overlap the requested frame", () => {
    const tracks = [
      track(1, 0, 29),
      track(2, 30, 90),
      track(3, 60, 120),
      { ...track(4, 30, 90), active: false },
    ];
    const index = buildTrackFrameIndex(tracks, 30);

    expect(tracksAtFrame(index, 15).map((item) => item.track_id)).toEqual([
      "track-1",
    ]);
    expect(tracksAtFrame(index, 75).map((item) => item.track_id)).toEqual([
      "track-2",
      "track-3",
    ]);
  });

  it("keeps large track collections bounded around the selection", () => {
    const tracks = Array.from({ length: 5_000 }, (_, index) =>
      track(index, index, index + 2),
    );
    const window = boundedTrackWindow(tracks, "track-2500", 60);

    expect(window).toHaveLength(60);
    expect(window.some((item) => item.track_id === "track-2500")).toBe(true);
    expect(window[0]!.track_id).toBe("track-2470");
  });

  it("samples keyframe markers while retaining both endpoints", () => {
    const keyframes = Array.from(
      { length: 1_000 },
      (_, frame_index): ReviewKeyframe => ({
        frame_index,
        timestamp_ms: frame_index * 40,
        bbox: { x1: 0, y1: 0, x2: 1, y2: 1 },
        locked: false,
      }),
    );
    const sampled = sampleKeyframes(keyframes, 32);

    expect(sampled).toHaveLength(32);
    expect(sampled[0]!.frame_index).toBe(0);
    expect(sampled.at(-1)!.frame_index).toBe(999);
  });
});
