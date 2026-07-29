export function processingPollDelay(unchangedPolls: number): number {
  const count = Math.max(0, Math.floor(unchangedPolls));
  return Math.min(5_000, 2_000 + count * 750);
}
