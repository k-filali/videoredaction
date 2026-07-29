import { useEffect } from "react";

import { Icon } from "./Icon";

export interface ToastMessage {
  id: number;
  tone: "success" | "error" | "info";
  message: string;
}
interface ToastProps {
  toast: ToastMessage;
  onDismiss: (id: number) => void;
}

export function Toast({ toast, onDismiss }: ToastProps) {
  useEffect(() => {
    const timer = window.setTimeout(() => onDismiss(toast.id), 4200);
    return () => window.clearTimeout(timer);
  }, [onDismiss, toast.id]);

  return (
    <div className={`toast toast-${toast.tone}`} role={toast.tone === "error" ? "alert" : "status"}>
      <span className="toast-icon">
        <Icon
          name={toast.tone === "success" ? "check" : toast.tone === "error" ? "warning" : "info"}
          size={16}
        />
      </span>
      <span>{toast.message}</span>
      <button type="button" onClick={() => onDismiss(toast.id)} aria-label="Dismiss notification">
        <Icon name="x" size={15} />
      </button>
    </div>
  );
}
