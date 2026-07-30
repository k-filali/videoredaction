/**
 * Detection kits offered to reviewers.
 *
 * Licence plates are the product vertical, so only that kit runs by default.
 * Face detection stays available for reviewers who opt in. This is the single
 * source of truth: every caller that starts detection must use it, otherwise
 * the API falls back to running every enabled detector.
 */
export interface DetectionKitOption {
  id: string;
  label: string;
  enabledByDefault: boolean;
}

export const detectionKitOptions: DetectionKitOption[] = [
  { id: "yolov9t-plate", label: "License plates", enabledByDefault: true },
  { id: "yunet-face", label: "Faces", enabledByDefault: false },
];

export const defaultDetectionKits: string[] = detectionKitOptions
  .filter((option) => option.enabledByDefault)
  .map((option) => option.id);
