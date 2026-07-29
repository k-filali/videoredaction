import { describe, expect, it, vi } from "vitest";

import {
  RESUMABLE_CHUNK_SIZE_BYTES,
  committedOffset,
  uploadFileResumable,
} from "./resumableUpload";

function videoFile(size: number): File {
  const blob = new Blob([new Uint8Array(size)], { type: "video/mp4" });
  return Object.assign(blob, { name: "incident.mp4", lastModified: 0 }) as File;
}

function incomplete(range?: string): Response {
  return new Response(null, {
    status: 308,
    headers: range ? { Range: range } : undefined,
  });
}

describe("resumable cloud uploads", () => {
  it("uploads in fixed 8 MiB chunks and reports committed progress", async () => {
    const file = videoFile(RESUMABLE_CHUNK_SIZE_BYTES * 2 + 1);
    const fetcher = vi
      .fn()
      .mockResolvedValueOnce(
        incomplete(`bytes=0-${RESUMABLE_CHUNK_SIZE_BYTES - 1}`),
      )
      .mockResolvedValueOnce(
        incomplete(`bytes=0-${RESUMABLE_CHUNK_SIZE_BYTES * 2 - 1}`),
      )
      .mockResolvedValueOnce(new Response(null, { status: 200 }));
    const progress: number[] = [];

    await uploadFileResumable(
      file,
      "https://upload.example/session",
      (value) => progress.push(value),
      { fetcher },
    );

    expect(fetcher).toHaveBeenCalledTimes(3);
    expect(fetcher.mock.calls.map((call) => call[1]?.headers)).toEqual([
      {
        "Content-Type": "video/mp4",
        "Content-Range": `bytes 0-${RESUMABLE_CHUNK_SIZE_BYTES - 1}/${file.size}`,
      },
      {
        "Content-Type": "video/mp4",
        "Content-Range": `bytes ${RESUMABLE_CHUNK_SIZE_BYTES}-${RESUMABLE_CHUNK_SIZE_BYTES * 2 - 1}/${file.size}`,
      },
      {
        "Content-Type": "video/mp4",
        "Content-Range": `bytes ${RESUMABLE_CHUNK_SIZE_BYTES * 2}-${file.size - 1}/${file.size}`,
      },
    ]);
    expect(
      fetcher.mock.calls.map((call) => (call[1]?.body as Blob).size),
    ).toEqual([RESUMABLE_CHUNK_SIZE_BYTES, RESUMABLE_CHUNK_SIZE_BYTES, 1]);
    expect(progress).toEqual([
      RESUMABLE_CHUNK_SIZE_BYTES / file.size,
      (RESUMABLE_CHUNK_SIZE_BYTES * 2) / file.size,
      1,
    ]);
  });

  it("queries the session offset after a retryable response", async () => {
    const file = videoFile(1024);
    const fetcher = vi
      .fn()
      .mockResolvedValueOnce(new Response(null, { status: 503 }))
      .mockResolvedValueOnce(incomplete())
      .mockResolvedValueOnce(new Response(null, { status: 200 }));
    const sleep = vi.fn().mockResolvedValue(undefined);

    await uploadFileResumable(file, "https://upload.example/session", vi.fn(), {
      fetcher,
      sleep,
    });

    expect(fetcher).toHaveBeenCalledTimes(3);
    expect(fetcher.mock.calls[1]?.[1]).toMatchObject({
      method: "PUT",
      headers: { "Content-Range": `bytes */${file.size}` },
    });
    expect((fetcher.mock.calls[1]?.[1]?.body as Blob).size).toBe(0);
    expect(sleep).toHaveBeenCalledWith(500);
  });

  it("finishes when recovery discovers the upload already committed", async () => {
    const file = videoFile(1024);
    const fetcher = vi
      .fn()
      .mockRejectedValueOnce(new TypeError("connection reset"))
      .mockResolvedValueOnce(new Response(null, { status: 200 }));
    const progress = vi.fn();

    await uploadFileResumable(
      file,
      "https://upload.example/session",
      progress,
      { fetcher, sleep: async () => undefined },
    );

    expect(fetcher).toHaveBeenCalledTimes(2);
    expect(progress).toHaveBeenLastCalledWith(1);
  });

  it("does not retry non-retryable upload failures", async () => {
    const fetcher = vi
      .fn()
      .mockResolvedValue(new Response(null, { status: 400 }));

    await expect(
      uploadFileResumable(
        videoFile(1024),
        "https://upload.example/session",
        vi.fn(),
        { fetcher },
      ),
    ).rejects.toMatchObject({ status: 400 });
    expect(fetcher).toHaveBeenCalledTimes(1);
  });

  it("bounds recovery retries with exponential backoff", async () => {
    const fetcher = vi.fn().mockRejectedValue(new TypeError("offline"));
    const sleep = vi.fn().mockResolvedValue(undefined);

    await expect(
      uploadFileResumable(
        videoFile(1024),
        "https://upload.example/session",
        vi.fn(),
        {
          fetcher,
          sleep,
          maxRetries: 3,
          baseRetryDelayMs: 100,
          maxRetryDelayMs: 250,
        },
      ),
    ).rejects.toThrow("offline");

    expect(fetcher).toHaveBeenCalledTimes(4);
    expect(sleep.mock.calls.map(([delay]) => delay)).toEqual([100, 200, 250]);
  });

  it("bounds retries when offset queries succeed without progress", async () => {
    const fetcher = vi
      .fn()
      .mockResolvedValueOnce(new Response(null, { status: 503 }))
      .mockResolvedValueOnce(incomplete())
      .mockResolvedValueOnce(new Response(null, { status: 503 }))
      .mockResolvedValueOnce(incomplete())
      .mockResolvedValueOnce(new Response(null, { status: 503 }));
    const sleep = vi.fn().mockResolvedValue(undefined);

    await expect(
      uploadFileResumable(
        videoFile(1024),
        "https://upload.example/session",
        vi.fn(),
        { fetcher, sleep, maxRetries: 2 },
      ),
    ).rejects.toMatchObject({ status: 503 });

    expect(fetcher).toHaveBeenCalledTimes(5);
    expect(sleep.mock.calls.map(([delay]) => delay)).toEqual([500, 1_000]);
  });

  it("validates committed byte ranges", () => {
    expect(
      committedOffset(incomplete("bytes=0-99"), 200, 0),
    ).toBe(100);
    expect(() =>
      committedOffset(incomplete("bytes=20-99"), 200, 0),
    ).toThrow("invalid byte range");
    expect(() =>
      committedOffset(incomplete("bytes=0-999"), 200, 0),
    ).toThrow("invalid byte offset");
  });
});
