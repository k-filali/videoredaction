import { afterEach, describe, expect, it, vi } from "vitest";

import { api, uploadVideo } from "./api";
import { RESUMABLE_CHUNK_SIZE_BYTES } from "./lib/resumableUpload";
import type { UploadAccepted } from "./types";

function videoFile(size = 1024): File {
  const blob = new Blob([new Uint8Array(size)], { type: "video/mp4" });
  return Object.assign(blob, { name: "incident.mp4", lastModified: 0 }) as File;
}

const accepted = {
  video: { id: "video-id" },
  job: { id: "job-id" },
} as unknown as UploadAccepted;

afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

describe("video upload transport selection", () => {
  it("uses the direct resumable session when the backend enables it", async () => {
    vi.spyOn(api, "getUploadCapability").mockResolvedValue({
      mode: "resumable",
      chunk_size_bytes: RESUMABLE_CHUNK_SIZE_BYTES,
      max_upload_bytes: 100 * 1024 * 1024,
    });
    vi.spyOn(api, "initiateUpload").mockResolvedValue({
      upload_id: "upload-id",
      session_url: "https://upload.example/session",
      chunk_size_bytes: RESUMABLE_CHUNK_SIZE_BYTES,
    });
    const complete = vi
      .spyOn(api, "completeUpload")
      .mockResolvedValue(accepted);
    const fetcher = vi
      .fn()
      .mockResolvedValue(new Response(null, { status: 200 }));
    vi.stubGlobal("fetch", fetcher);
    const progress = vi.fn();

    const result = await uploadVideo(videoFile(), progress);

    expect(result).toBe(accepted);
    expect(fetcher).toHaveBeenCalledWith(
      "https://upload.example/session",
      expect.objectContaining({
        method: "PUT",
        headers: {
          "Content-Type": "video/mp4",
          "Content-Range": "bytes 0-1023/1024",
        },
      }),
    );
    expect(complete).toHaveBeenCalledWith("upload-id");
    expect(progress).toHaveBeenLastCalledWith(1);
  });

  it("retains multipart XHR upload for local storage", async () => {
    vi.spyOn(api, "getUploadCapability").mockResolvedValue({
      mode: "multipart",
    });
    vi.stubGlobal("sessionStorage", {
      getItem: () => "review-test",
      setItem: vi.fn(),
    });
    let progressListener: ((event: {
      lengthComputable: boolean;
      loaded: number;
      total: number;
    }) => void) | undefined;
    let loadListener: (() => void) | undefined;
    const send = vi.fn(function sendMultipart(this: FakeXMLHttpRequest) {
      progressListener?.({ lengthComputable: true, loaded: 10, total: 10 });
      loadListener?.();
      expect(this.response).toBe(accepted);
    });

    class FakeXMLHttpRequest {
      response = accepted;
      responseType = "";
      status = 202;
      upload = {
        addEventListener: (
          type: string,
          listener: typeof progressListener,
        ) => {
          if (type === "progress") progressListener = listener;
        },
      };

      open(): void {}

      setRequestHeader(): void {}

      addEventListener(type: string, listener: () => void): void {
        if (type === "load") loadListener = listener;
      }

      send = send;
    }

    vi.stubGlobal("XMLHttpRequest", FakeXMLHttpRequest);
    const progress = vi.fn();

    const result = await uploadVideo(videoFile(), progress);

    expect(result).toBe(accepted);
    expect(send).toHaveBeenCalledOnce();
    expect(progress).toHaveBeenLastCalledWith(1);
  });
});
