import { useCallback, useEffect, useRef, useState } from "react";

import { Dashboard } from "./components/Dashboard";
import { ReviewWorkspace } from "./components/ReviewWorkspace";
import { Toast, type ToastMessage } from "./components/Toast";
import type { VideoAsset } from "./types";

function videoFromLocation(): string | null {
  return new URLSearchParams(window.location.search).get("video");
}

function App() {
  const [videoId, setVideoId] = useState<string | null>(videoFromLocation);
  const [toasts, setToasts] = useState<ToastMessage[]>([]);
  const toastId = useRef(0);

  useEffect(() => {
    const handlePopState = () => setVideoId(videoFromLocation());
    window.addEventListener("popstate", handlePopState);
    return () => window.removeEventListener("popstate", handlePopState);
  }, []);

  useEffect(() => {
    document.title = videoId ? "Review evidence · ClearFrame" : "ClearFrame · Video evidence";
  }, [videoId]);

  const notify = useCallback(
    (message: string, tone: "success" | "error" | "info" = "info") => {
      toastId.current += 1;
      setToasts((current) => [...current.slice(-2), { id: toastId.current, message, tone }]);
    },
    [],
  );

  const openVideo = useCallback((video: VideoAsset) => {
    const next = `/?video=${encodeURIComponent(video.id)}`;
    window.history.pushState({ videoId: video.id }, "", next);
    setVideoId(video.id);
    window.scrollTo(0, 0);
  }, []);

  const returnHome = useCallback(() => {
    window.history.pushState({}, "", "/");
    setVideoId(null);
    window.scrollTo(0, 0);
  }, []);

  return (
    <>
      {videoId ? (
        <ReviewWorkspace videoId={videoId} onBack={returnHome} onNotify={notify} />
      ) : (
        <Dashboard onOpenVideo={openVideo} onNotify={notify} />
      )}
      <div className="toast-stack" aria-live="polite">
        {toasts.map((toast) => (
          <Toast
            toast={toast}
            key={toast.id}
            onDismiss={(id) => setToasts((current) => current.filter((item) => item.id !== id))}
          />
        ))}
      </div>
    </>
  );
}

export default App;

