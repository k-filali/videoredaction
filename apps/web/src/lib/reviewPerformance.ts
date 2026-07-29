import type { ReviewKeyframe, ReviewTrack } from "../types";

export interface TrackFrameIndex {
  bucketSizeFrames: number;
  buckets: Map<number, ReviewTrack[]>;
}

export function buildTrackFrameIndex(
  tracks: ReviewTrack[],
  bucketSizeFrames: number,
): TrackFrameIndex {
  const size = Math.max(1, Math.round(bucketSizeFrames));
  const buckets = new Map<number, ReviewTrack[]>();

  tracks.forEach((track) => {
    if (!track.active || track.end_frame < track.start_frame) return;
    const firstBucket = Math.floor(Math.max(0, track.start_frame) / size);
    const lastBucket = Math.floor(Math.max(0, track.end_frame) / size);
    for (let bucket = firstBucket; bucket <= lastBucket; bucket += 1) {
      const entries = buckets.get(bucket);
      if (entries) entries.push(track);
      else buckets.set(bucket, [track]);
    }
  });

  return { bucketSizeFrames: size, buckets };
}

export function tracksAtFrame(index: TrackFrameIndex, frame: number): ReviewTrack[] {
  if (frame < 0) return [];
  const candidates =
    index.buckets.get(Math.floor(frame / index.bucketSizeFrames)) ?? [];
  return candidates.filter(
    (track) =>
      track.active && frame >= track.start_frame && frame <= track.end_frame,
  );
}

export function boundedTrackWindow(
  tracks: ReviewTrack[],
  selectedTrackId: string | null,
  limit: number,
): ReviewTrack[] {
  const windowSize = Math.max(1, Math.floor(limit));
  if (tracks.length <= windowSize) return tracks;

  const selectedIndex = Math.max(
    0,
    tracks.findIndex((track) => track.track_id === selectedTrackId),
  );
  const start = Math.min(
    tracks.length - windowSize,
    Math.max(0, selectedIndex - Math.floor(windowSize / 2)),
  );
  return tracks.slice(start, start + windowSize);
}

export function sampleKeyframes(
  keyframes: ReviewKeyframe[],
  limit: number,
): ReviewKeyframe[] {
  const sampleSize = Math.max(2, Math.floor(limit));
  if (keyframes.length <= sampleSize) return keyframes;

  const sampled: ReviewKeyframe[] = [];
  const lastIndex = keyframes.length - 1;
  for (let index = 0; index < sampleSize; index += 1) {
    sampled.push(
      keyframes[Math.round((index / (sampleSize - 1)) * lastIndex)]!,
    );
  }
  return sampled;
}
