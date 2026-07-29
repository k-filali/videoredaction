import { useEffect, useRef, useState } from "react";

import { defaultShapeFor } from "../lib/redaction";
import type {
  ClassTreatment,
  ExportArtifact,
  RedactionShape,
  RedactionStyle,
} from "../types";
import { Icon } from "./Icon";

interface ExportModalProps {
  open: boolean;
  unresolvedCount: number;
  pendingSuggestionCount: number;
  warnings: string[];
  audioPresent: boolean;
  revision: number;
  style: RedactionStyle;
  classes: string[];
  classTreatments: Record<string, ClassTreatment>;
  exporting: boolean;
  artifact: ExportArtifact | null;
  pollError: string | null;
  onStyleChange: (style: RedactionStyle) => void;
  onClassTreatmentsChange: (next: Record<string, ClassTreatment>) => void;
  onExport: () => void;
  onRetryStatus: () => void;
  onClose: () => void;
}

const styles: { value: RedactionStyle; title: string; detail: string }[] = [
  { value: "pixelate", title: "Pixelate", detail: "Clear privacy masking" },
  { value: "gaussian_blur", title: "Gaussian blur", detail: "Softer visual treatment" },
  { value: "black_box", title: "Black box", detail: "Maximum visual coverage" },
];

const presets: {
  id: string;
  title: string;
  detail: string;
  style: RedactionStyle;
  treatments: Record<string, ClassTreatment>;
}[] = [
  {
    id: "standard",
    title: "Standard disclosure",
    detail: "Pixelated regions, elliptical faces",
    style: "pixelate",
    treatments: { face: { shape: "ellipse" } },
  },
  {
    id: "sensitive",
    title: "Sensitive victim",
    detail: "Blacked-out plates for harassment or IPV files",
    style: "pixelate",
    treatments: {
      face: { shape: "ellipse" },
      license_plate: { style: "black_box" },
    },
  },
  {
    id: "protected",
    title: "Protected technique",
    detail: "Blackout everything at maximum coverage",
    style: "black_box",
    treatments: {},
  },
];

function classLabel(value: string): string {
  if (value === "license_plate") return "License plates";
  const spaced = value.replaceAll("_", " ");
  return `${spaced.charAt(0).toUpperCase()}${spaced.slice(1)}s`;
}

function effectiveShape(
  className: string,
  treatments: Record<string, ClassTreatment>,
): RedactionShape {
  return treatments[className]?.shape ?? defaultShapeFor(className);
}

export function ExportModal({
  open,
  unresolvedCount,
  pendingSuggestionCount,
  warnings,
  audioPresent,
  revision,
  style,
  classes,
  classTreatments,
  exporting,
  artifact,
  pollError,
  onStyleChange,
  onClassTreatmentsChange,
  onExport,
  onRetryStatus,
  onClose,
}: ExportModalProps) {
  const closeRef = useRef<HTMLButtonElement>(null);
  const [audioAcknowledged, setAudioAcknowledged] = useState(false);

  useEffect(() => {
    if (!open) return;
    closeRef.current?.focus();
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape" && !exporting) onClose();
    };
    window.addEventListener("keydown", closeOnEscape);
    return () => window.removeEventListener("keydown", closeOnEscape);
  }, [exporting, onClose, open]);

  if (!open) return null;
  const complete = artifact?.status === "COMPLETED";
  const failed = artifact?.status === "FAILED";
  const interrupted = Boolean(pollError && artifact && !complete && !failed);
  const pendingCount = unresolvedCount + pendingSuggestionCount;
  const progress =
    artifact?.status === "VERIFYING"
      ? 88
      : artifact?.status === "RENDERING"
        ? 52
        : artifact?.status === "QUEUED"
          ? 12
          : complete
            ? 100
            : 0;

  return (
    <div className="modal-backdrop" role="presentation" onMouseDown={!exporting ? onClose : undefined}>
      <section
        className="export-modal"
        role="dialog"
        aria-modal="true"
        aria-labelledby="export-title"
        onMouseDown={(event) => event.stopPropagation()}
      >
        <header className="modal-header">
          <div>
            <span className="drawer-eyebrow">Derived copy</span>
            <h2 id="export-title">{complete ? "Export verified" : "Export redacted video"}</h2>
          </div>
          <button
            ref={closeRef}
            className="icon-button"
            type="button"
            onClick={onClose}
            disabled={exporting && !complete && !failed}
            aria-label="Close export dialog"
          >
            <Icon name="x" />
          </button>
        </header>

        {complete ? (
          <div className="export-complete">
            <span className="complete-mark"><Icon name="check" size={28} /></span>
            <h3>Your redacted copy is ready</h3>
            <p>The output was rendered from frozen revision {artifact.review_revision} and verified.</p>
            {warnings.length > 0 && (
              <ul className="export-warning-list">
                {warnings.map((warning) => (
                  <li key={warning}>
                    <Icon name="warning" size={13} />
                    {warning}
                  </li>
                ))}
              </ul>
            )}
            {audioPresent && (
              <p className="audio-export-warning">
                Audio content was carried into the export without review or redaction and may be
                re-encoded.
              </p>
            )}
            <div className="hash-card">
              <span>SHA-256 checksum</span>
              <code>{artifact.export_sha256 ?? "Checksum recorded in manifest"}</code>
            </div>
            <div className="modal-actions split">
              {artifact.manifest_url && (
                <a className="button button-secondary" href={artifact.manifest_url} target="_blank" rel="noreferrer">
                  <Icon name="shield" size={16} />
                  View manifest
                </a>
              )}
              {artifact.download_url && (
                <a className="button button-primary" href={artifact.download_url}>
                  <Icon name="download" size={16} />
                  Download MP4
                </a>
              )}
            </div>
          </div>
        ) : exporting || artifact ? (
          <div className="export-progress-view">
            {interrupted ? (
              <>
                <span className="export-failed-mark"><Icon name="warning" size={24} /></span>
                <h3>Export status is unavailable</h3>
                <p>{pollError}</p>
                <div className="modal-actions split">
                  <button className="button button-ghost" type="button" onClick={onClose}>
                    Return to review
                  </button>
                  <button className="button button-secondary" type="button" onClick={onRetryStatus}>
                    Retry status
                  </button>
                </div>
              </>
            ) : failed ? (
              <>
                <span className="export-failed-mark"><Icon name="warning" size={24} /></span>
                <h3>Export could not be completed</h3>
                <p>{artifact?.error_message ?? "The renderer reported an unexpected error."}</p>
                <button className="button button-secondary" type="button" onClick={onClose}>
                  Return to review
                </button>
              </>
            ) : (
              <>
                <span className="export-spinner"><span className="spinner" /></span>
                <h3>
                  {artifact?.status === "VERIFYING" ? "Verifying derived copy" : "Rendering redactions"}
                </h3>
                <p>Keep this workspace open. The immutable source remains untouched.</p>
                <div className="export-progress-track">
                  <span style={{ width: `${progress}%` }} />
                </div>
                <span className="export-stage">
                  {artifact?.status === "VERIFYING" ? "Checking duration, dimensions and hashes" : "Applying approved track regions"}
                </span>
              </>
            )}
          </div>
        ) : (
          <>
            <div className="export-readiness">
              <span className={pendingCount === 0 ? "ready" : "blocked"}>
                <Icon name={pendingCount === 0 ? "check" : "warning"} size={17} />
              </span>
              <div>
                <strong>
                  {pendingCount === 0
                    ? "Review is ready to export"
                    : "Review decisions are still pending"}
                </strong>
                <p>
                  {pendingCount === 0
                    ? `Revision ${revision} will be frozen for this render.`
                    : [
                        unresolvedCount > 0
                          ? `${unresolvedCount} unconfirmed track${
                              unresolvedCount === 1 ? "" : "s"
                            }`
                          : "",
                        pendingSuggestionCount > 0
                          ? `${pendingSuggestionCount} pending context suggestion${
                              pendingSuggestionCount === 1 ? "" : "s"
                            }`
                          : "",
                      ]
                        .filter(Boolean)
                        .join(" and ") +
                      " will be recorded as warnings in the export manifest. " +
                      "Detected tracks are redacted by default."}
                </p>
              </div>
            </div>

            <fieldset className="preset-options">
              <legend>Case preset</legend>
              <div className="preset-row">
                {presets.map((preset) => (
                  <button
                    className="preset-chip"
                    type="button"
                    key={preset.id}
                    title={preset.detail}
                    onClick={() => {
                      onStyleChange(preset.style);
                      onClassTreatmentsChange({ ...preset.treatments });
                    }}
                  >
                    <strong>{preset.title}</strong>
                    <small>{preset.detail}</small>
                  </button>
                ))}
              </div>
            </fieldset>

            <fieldset className="style-options">
              <legend>Default redaction treatment</legend>
              {styles.map((option) => (
                <label className={option.value === style ? "is-selected" : ""} key={option.value}>
                  <input
                    type="radio"
                    name="redaction-style"
                    value={option.value}
                    checked={option.value === style}
                    onChange={() => onStyleChange(option.value)}
                  />
                  <span className={`style-swatch swatch-${option.value}`} />
                  <span>
                    <strong>{option.title}</strong>
                    <small>{option.detail}</small>
                  </span>
                  <i><Icon name="check" size={14} /></i>
                </label>
              ))}
            </fieldset>

            {classes.length > 0 && (
              <fieldset className="class-treatment-options">
                <legend>Per-class overrides</legend>
                {classes.map((className) => (
                  <div className="class-treatment-row" key={className}>
                    <span>{classLabel(className)}</span>
                    <label>
                      <span className="visually-hidden">
                        {classLabel(className)} treatment
                      </span>
                      <select
                        value={classTreatments[className]?.style ?? ""}
                        onChange={(event) =>
                          onClassTreatmentsChange({
                            ...classTreatments,
                            [className]: {
                              ...classTreatments[className],
                              style: (event.target.value || undefined) as
                                | RedactionStyle
                                | undefined,
                            },
                          })
                        }
                      >
                        <option value="">Use default</option>
                        {styles.map((option) => (
                          <option value={option.value} key={option.value}>
                            {option.title}
                          </option>
                        ))}
                      </select>
                    </label>
                    <label>
                      <span className="visually-hidden">
                        {classLabel(className)} shape
                      </span>
                      <select
                        value={effectiveShape(className, classTreatments)}
                        onChange={(event) =>
                          onClassTreatmentsChange({
                            ...classTreatments,
                            [className]: {
                              ...classTreatments[className],
                              shape: event.target.value as RedactionShape,
                            },
                          })
                        }
                      >
                        <option value="rectangle">Rectangle</option>
                        <option value="ellipse">Ellipse</option>
                      </select>
                    </label>
                  </div>
                ))}
              </fieldset>
            )}

            <div className="export-safety-note">
              <Icon name="shield" size={17} />
              <span>
                ClearFrame creates a new file, manifest, and checksum. The original is never
                modified.
              </span>
            </div>

            {audioPresent && (
              <label className="audio-acknowledgement">
                <input
                  type="checkbox"
                  checked={audioAcknowledged}
                  onChange={(event) => setAudioAcknowledged(event.target.checked)}
                />
                <span>
                  <strong>Audio is not redacted</strong>
                  This video-only MVP does not detect or redact spoken names or other
                  sensitive audio. The audio stream may be re-encoded.
                </span>
              </label>
            )}

            <div className="modal-actions">
              <button className="button button-ghost" type="button" onClick={onClose}>
                Cancel
              </button>
              <button
                className="button button-primary"
                type="button"
                disabled={audioPresent && !audioAcknowledged}
                onClick={onExport}
              >
                <Icon name="download" size={16} />
                Render derived copy
              </button>
            </div>
          </>
        )}
      </section>
    </div>
  );
}
