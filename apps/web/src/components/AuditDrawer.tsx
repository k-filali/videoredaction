import { useEffect, useRef } from "react";

import { formatTime } from "../lib/geometry";
import type { AuditAction, AuditLog } from "../types";
import { Icon, type IconName } from "./Icon";

interface AuditDrawerProps {
  open: boolean;
  loading: boolean;
  audit: AuditLog | null;
  onClose: () => void;
  onRefresh: () => void;
}
const actionCopy: Record<
  AuditAction["action_type"],
  { title: string; icon: IconName; tone: string }
> = {
  ACCEPT_PROPOSAL: { title: "Proposal accepted", icon: "check", tone: "success" },
  RESTORE_TRACK: { title: "Region restored", icon: "eye", tone: "neutral" },
  REDACT_TRACK: { title: "Redaction enabled", icon: "eye-off", tone: "success" },
  DELETE_FALSE_POSITIVE: { title: "False positive removed", icon: "trash", tone: "neutral" },
  CREATE_MANUAL_REGION: { title: "Manual region added", icon: "plus", tone: "success" },
  RESIZE_REGION: { title: "Region resized", icon: "zoom-in", tone: "neutral" },
  MOVE_REGION: { title: "Region moved", icon: "layers", tone: "neutral" },
  SPLIT_TRACK: { title: "Track split", icon: "layers", tone: "neutral" },
  MERGE_TRACKS: { title: "Tracks merged", icon: "layers", tone: "neutral" },
  EXTEND_TRACK: { title: "Track extended", icon: "clock", tone: "neutral" },
  TRIM_TRACK: { title: "Track trimmed", icon: "clock", tone: "neutral" },
  CHANGE_CLASS: { title: "Class changed", icon: "filter", tone: "neutral" },
  UNDO: { title: "Change undone", icon: "undo", tone: "neutral" },
  REDO: { title: "Change reapplied", icon: "redo", tone: "neutral" },
  EXPORT_REQUESTED: { title: "Export requested", icon: "download", tone: "export" },
  EXPORT_COMPLETED: { title: "Export verified", icon: "shield", tone: "export" },
};

function actionDetail(action: AuditAction): string {
  if (action.timestamp_ms !== null) return `at ${formatTime(action.timestamp_ms)}`;
  if (action.track_id) return `track ${action.track_id.slice(0, 8)}`;
  return `revision ${action.revision}`;
}

export function AuditDrawer({ open, loading, audit, onClose, onRefresh }: AuditDrawerProps) {
  const closeRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    if (!open) return;
    closeRef.current?.focus();
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    window.addEventListener("keydown", closeOnEscape);
    return () => window.removeEventListener("keydown", closeOnEscape);
  }, [onClose, open]);

  if (!open) return null;

  return (
    <div className="drawer-backdrop" role="presentation" onMouseDown={onClose}>
      <aside
        className="audit-drawer"
        role="dialog"
        aria-modal="true"
        aria-labelledby="audit-title"
        onMouseDown={(event) => event.stopPropagation()}
      >
        <header className="drawer-header">
          <div>
            <span className="drawer-eyebrow">Append-only record</span>
            <h2 id="audit-title">Review history</h2>
          </div>
          <button
            ref={closeRef}
            className="icon-button"
            type="button"
            onClick={onClose}
            aria-label="Close review history"
          >
            <Icon name="x" />
          </button>
        </header>

        <div className="audit-integrity">
          <span><Icon name="shield" size={17} /></span>
          <div>
            <strong>Audit trail protected</strong>
            <p>Every reviewer and export action is recorded with its revision.</p>
          </div>
        </div>

        <div className="drawer-summary">
          <span>
            <strong>{audit?.actions.length ?? 0}</strong>
            events
          </span>
          <span>
            <strong>{audit?.revision ?? 0}</strong>
            current revision
          </span>
          <button type="button" onClick={onRefresh} disabled={loading}>
            Refresh
          </button>
        </div>

        <div className="audit-list">
          {loading ? (
            <div className="drawer-loading">
              <span className="spinner" />
              Loading history
            </div>
          ) : !audit || audit.actions.length === 0 ? (
            <div className="drawer-empty">
              <Icon name="history" size={25} />
              <h3>No decisions recorded yet</h3>
              <p>Reviewer actions will appear here as they are made.</p>
            </div>
          ) : (
            [...audit.actions].reverse().map((action) => {
              const copy = actionCopy[action.action_type];
              return (
                <article className="audit-event" key={action.id}>
                  <span className={`audit-event-icon tone-${copy.tone}`}>
                    <Icon name={copy.icon} size={15} />
                  </span>
                  <div>
                    <div className="audit-event-heading">
                      <strong>{copy.title}</strong>
                      <span>r{action.revision}</span>
                    </div>
                    <p>
                      {actionDetail(action)} · {action.reviewer_session_id}
                    </p>
                    <time dateTime={action.created_at}>
                      {new Date(action.created_at).toLocaleString([], {
                        month: "short",
                        day: "numeric",
                        hour: "numeric",
                        minute: "2-digit",
                        second: "2-digit",
                      })}
                    </time>
                  </div>
                </article>
              );
            })
          )}
        </div>
      </aside>
    </div>
  );
}
