import type { NormalizedBox, ReviewKeyframe, ReviewTrack } from "../types";

const orderedKeyframeCache = new WeakMap<
  ReviewKeyframe[],
  readonly ReviewKeyframe[]
>();

function orderedKeyframes(keyframes: ReviewKeyframe[]): readonly ReviewKeyframe[] {
  const cached = orderedKeyframeCache.get(keyframes);
  if (cached) return cached;

  let ordered = true;
  for (let index = 1; index < keyframes.length; index += 1) {
    if (keyframes[index - 1]!.frame_index > keyframes[index]!.frame_index) {
      ordered = false;
      break;
    }
  }
  const result = ordered
    ? keyframes
    : [...keyframes].sort((a, b) => a.frame_index - b.frame_index);
  orderedKeyframeCache.set(keyframes, result);
  return result;
}

export function clamp(value: number, min = 0, max = 1): number {
  return Math.min(max, Math.max(min, value));
}
export function normalizeBox(a: { x: number; y: number }, b: { x: number; y: number }): NormalizedBox {
  return {
    x1: clamp(Math.min(a.x, b.x)),
    y1: clamp(Math.min(a.y, b.y)),
    x2: clamp(Math.max(a.x, b.x)),
    y2: clamp(Math.max(a.y, b.y)),
  };
}

export function isUsefulBox(box: NormalizedBox): boolean {
  return box.x2 - box.x1 >= 0.008 && box.y2 - box.y1 >= 0.008;
}

export function boxAtFrame(track: ReviewTrack, frame: number): NormalizedBox | null {
  if (!track.active || frame < track.start_frame || frame > track.end_frame) return null;
  const keyframes = orderedKeyframes(track.keyframes);
  if (keyframes.length === 0) return null;
  if (keyframes.length === 1 || frame <= keyframes[0]!.frame_index) return keyframes[0]!.bbox;
  const last = keyframes[keyframes.length - 1]!;
  if (frame >= last.frame_index) return last.bbox;

  let low = 0;
  let high = keyframes.length - 1;
  while (high - low > 1) {
    const middle = Math.floor((low + high) / 2);
    if (keyframes[middle]!.frame_index <= frame) low = middle;
    else high = middle;
  }
  const left: ReviewKeyframe = keyframes[low]!;
  const right: ReviewKeyframe = keyframes[high]!;
  const span = right.frame_index - left.frame_index;
  const t = span === 0 ? 0 : (frame - left.frame_index) / span;
  return {
    x1: left.bbox.x1 + (right.bbox.x1 - left.bbox.x1) * t,
    y1: left.bbox.y1 + (right.bbox.y1 - left.bbox.y1) * t,
    x2: left.bbox.x2 + (right.bbox.x2 - left.bbox.x2) * t,
    y2: left.bbox.y2 + (right.bbox.y2 - left.bbox.y2) * t,
  };
}

export function moveBox(box: NormalizedBox, dx: number, dy: number): NormalizedBox {
  const width = box.x2 - box.x1;
  const height = box.y2 - box.y1;
  const x1 = clamp(box.x1 + dx, 0, 1 - width);
  const y1 = clamp(box.y1 + dy, 0, 1 - height);
  return { x1, y1, x2: x1 + width, y2: y1 + height };
}

export function resizeBox(box: NormalizedBox, dx: number, dy: number): NormalizedBox {
  return {
    ...box,
    x2: clamp(box.x2 + dx, box.x1 + 0.008, 1),
    y2: clamp(box.y2 + dy, box.y1 + 0.008, 1),
  };
}

export function formatTime(milliseconds: number): string {
  const totalSeconds = Math.max(0, Math.floor(milliseconds / 1000));
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  const frames = Math.floor((milliseconds % 1000) / 10);
  return `${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}.${String(frames).padStart(2, "0")}`;
}

export function humanFileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  const units = ["KB", "MB", "GB"];
  let value = bytes / 1024;
  let unit = units[0]!;
  for (let index = 1; index < units.length && value >= 1024; index += 1) {
    value /= 1024;
    unit = units[index]!;
  }
  return `${value.toFixed(value >= 10 ? 0 : 1)} ${unit}`;
}
