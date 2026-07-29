export const RESUMABLE_CHUNK_SIZE_BYTES = 8 * 1024 * 1024;

const RETRYABLE_STATUS_CODES = new Set([408, 425, 429, 500, 502, 503, 504]);

type Fetcher = (
  input: RequestInfo | URL,
  init?: RequestInit,
) => Promise<Response>;

export interface ResumableUploadOptions {
  fetcher?: Fetcher;
  sleep?: (milliseconds: number) => Promise<void>;
  maxRetries?: number;
  baseRetryDelayMs?: number;
  maxRetryDelayMs?: number;
  signal?: AbortSignal;
}

export class ResumableUploadError extends Error {
  status: number;

  constructor(message: string, status = 0) {
    super(message);
    this.name = "ResumableUploadError";
    this.status = status;
  }
}

function retryDelay(
  attempt: number,
  baseDelayMs: number,
  maxDelayMs: number,
): number {
  return Math.min(maxDelayMs, baseDelayMs * 2 ** attempt);
}

function retryable(error: unknown): boolean {
  return (
    !(error instanceof ResumableUploadError) ||
    error.status === 0 ||
    RETRYABLE_STATUS_CODES.has(error.status)
  );
}

function errorForResponse(response: Response): ResumableUploadError {
  return new ResumableUploadError(
    `Cloud upload failed with status ${response.status}`,
    response.status,
  );
}

export function committedOffset(
  response: Response,
  totalBytes: number,
  fallback: number,
): number {
  const range = response.headers.get("Range");
  if (!range) return fallback;
  const match = /^bytes=0-(\d+)$/.exec(range.trim());
  if (!match) {
    throw new ResumableUploadError("Cloud upload returned an invalid byte range");
  }
  const end = Number(match[1]);
  const next = end + 1;
  if (!Number.isSafeInteger(next) || next < 0 || next > totalBytes) {
    throw new ResumableUploadError("Cloud upload returned an invalid byte offset");
  }
  return next;
}

async function queryCommittedOffset(
  sessionUrl: string,
  totalBytes: number,
  fetcher: Fetcher,
  signal?: AbortSignal,
): Promise<number> {
  const response = await fetcher(sessionUrl, {
    method: "PUT",
    headers: {
      "Content-Range": `bytes */${totalBytes}`,
    },
    body: new Blob([]),
    signal,
  });
  if (response.status === 308) {
    return committedOffset(response, totalBytes, 0);
  }
  if (response.ok) return totalBytes;
  throw errorForResponse(response);
}

async function recoverCommittedOffset(
  sessionUrl: string,
  totalBytes: number,
  options: Required<
    Pick<
      ResumableUploadOptions,
      "fetcher" | "sleep" | "maxRetries" | "baseRetryDelayMs" | "maxRetryDelayMs"
    >
  > &
    Pick<ResumableUploadOptions, "signal">,
  startingAttempt: number,
  initialError: unknown,
): Promise<{ offset: number; attempts: number }> {
  let lastError = initialError;
  for (
    let attempt = startingAttempt;
    attempt < options.maxRetries;
    attempt += 1
  ) {
    await options.sleep(
      retryDelay(attempt, options.baseRetryDelayMs, options.maxRetryDelayMs),
    );
    try {
      return {
        offset: await queryCommittedOffset(
          sessionUrl,
          totalBytes,
          options.fetcher,
          options.signal,
        ),
        attempts: attempt + 1,
      };
    } catch (error) {
      if (options.signal?.aborted) throw error;
      if (!retryable(error)) throw error;
      lastError = error;
    }
  }
  throw lastError instanceof Error
    ? lastError
    : new ResumableUploadError("Could not recover the cloud upload session");
}

export async function uploadFileResumable(
  file: File,
  sessionUrl: string,
  onProgress: (progress: number) => void,
  options: ResumableUploadOptions = {},
): Promise<void> {
  if (file.size <= 0) {
    throw new ResumableUploadError("Video is empty");
  }
  const resolved = {
    fetcher: options.fetcher ?? fetch,
    sleep:
      options.sleep ??
      ((milliseconds: number) =>
        new Promise<void>((resolve) => window.setTimeout(resolve, milliseconds))),
    maxRetries: options.maxRetries ?? 4,
    baseRetryDelayMs: options.baseRetryDelayMs ?? 500,
    maxRetryDelayMs: options.maxRetryDelayMs ?? 4_000,
    signal: options.signal,
  };
  if (resolved.maxRetries < 1) {
    throw new RangeError("maxRetries must be at least one");
  }

  let offset = 0;
  let retriesWithoutProgress = 0;
  while (offset < file.size) {
    const endExclusive = Math.min(
      offset + RESUMABLE_CHUNK_SIZE_BYTES,
      file.size,
    );
    const chunk = file.slice(offset, endExclusive);
    let response: Response;
    try {
      response = await resolved.fetcher(sessionUrl, {
        method: "PUT",
        headers: {
          "Content-Type": file.type || "application/octet-stream",
          "Content-Range": `bytes ${offset}-${endExclusive - 1}/${file.size}`,
        },
        body: chunk,
        signal: resolved.signal,
      });
    } catch (error) {
      if (resolved.signal?.aborted || !retryable(error)) throw error;
      const recovered = await recoverCommittedOffset(
        sessionUrl,
        file.size,
        resolved,
        retriesWithoutProgress,
        error,
      );
      if (recovered.offset < offset) {
        throw new ResumableUploadError(
          "Cloud upload session moved behind the local offset",
        );
      }
      retriesWithoutProgress =
        recovered.offset > offset ? 0 : recovered.attempts;
      offset = recovered.offset;
      onProgress(offset / file.size);
      continue;
    }

    if (response.status === 308) {
      const committed = committedOffset(response, file.size, endExclusive);
      if (committed <= offset) {
        throw new ResumableUploadError("Cloud upload made no forward progress");
      }
      offset = committed;
      retriesWithoutProgress = 0;
      onProgress(offset / file.size);
      continue;
    }
    if (response.ok) {
      offset = file.size;
      onProgress(1);
      break;
    }
    const responseError = errorForResponse(response);
    if (!retryable(responseError)) throw responseError;
    const recovered = await recoverCommittedOffset(
      sessionUrl,
      file.size,
      resolved,
      retriesWithoutProgress,
      responseError,
    );
    if (recovered.offset < offset) {
      throw new ResumableUploadError(
        "Cloud upload session moved behind the local offset",
      );
    }
    retriesWithoutProgress =
      recovered.offset > offset ? 0 : recovered.attempts;
    offset = recovered.offset;
    onProgress(offset / file.size);
  }
}
