import { describe, expect, it } from "vitest";

import type { ReviewTrack } from "../types";
import { boxAtFrame, moveBox, normalizeBox, resizeBox } from "./geometry";

const track: ReviewTrack = {
  track_id: "track-1",
  class_name: "license_plate",
  source: "MODEL",
  active: true,
  redacted: true,
  accepted: false,
  start_frame: 0,
  end_frame: 20,
  start_ms: 0,
  end_ms: 800,
  confidence: 0.9,
  warning: null,
  keyframes: [
    {
      frame_index: 0,
      timestamp_ms: 0,
      bbox: { x1: 0.1, y1: 0.2, x2: 0.3, y2: 0.4 },
      locked: true,
    },
    {
      frame_index: 20,
      timestamp_ms: 800,
      bbox: { x1: 0.3, y1: 0.4, x2: 0.5, y2: 0.6 },
      locked: true,
    },
  ],
};

describe("review geometry", () => {
  it("interpolates track boxes", () => {
    expect(boxAtFrame(track, 10)).toEqual({
      x1: 0.2,
      y1: 0.30000000000000004,
      x2: 0.4,
      y2: 0.5,
    });
  });

  it("normalizes reverse drag coordinates", () => {
    expect(normalizeBox({ x: 0.8, y: 0.7 }, { x: 0.2, y: 0.1 })).toEqual({
      x1: 0.2,
      y1: 0.1,
      x2: 0.8,
      y2: 0.7,
    });
  });

  it("keeps moved and resized boxes within frame bounds", () => {
    expect(moveBox({ x1: 0.8, y1: 0.8, x2: 1, y2: 1 }, 0.4, 0.4)).toEqual({
      x1: 0.8,
      y1: 0.8,
      x2: 1,
      y2: 1,
    });
    expect(resizeBox({ x1: 0.2, y1: 0.2, x2: 0.3, y2: 0.3 }, -1, -1)).toEqual({
      x1: 0.2,
      y1: 0.2,
      x2: 0.20800000000000002,
      y2: 0.20800000000000002,
    });
  });
});
