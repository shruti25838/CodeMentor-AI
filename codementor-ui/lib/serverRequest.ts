/**
 * Fetch wrapper for a backend that may be asleep (free tier, up to a minute to wake).
 *
 * - After SLOW_AFTER_MS without a response, the shared status becomes "waking" so the UI can say so, but only
 *   until the server has answered any request in this page session. After that, a slow request is real work
 *   (indexing, an LLM answer) and the caller shows its own progress text.
 * - Only requests marked `retry: true` (safe reads) are retried, on network errors or 502/503/504.
 *   Requests that start work, such as indexing a repository, are never retried.
 * - When the server cannot be reached, the status becomes "unreachable" and the call throws a plain-English error.
 */

export const WAKING_MESSAGE = "Waking up the server, this can take up to a minute";
export const UNREACHABLE_MESSAGE =
    "Could not reach the server. It may still be starting up. Please wait a minute and try again.";

export const SLOW_AFTER_MS = 4000;
export const READ_DEADLINE_MS = 75_000;
const MAX_RETRY_DELAY_MS = 8000;
const WAKING_STATUSES = new Set([502, 503, 504]);

export type ServerStatus = "ok" | "waking" | "unreachable";

let slowRequests = 0;
let unreachable = false;
let serverHasAnswered = false;
const listeners = new Set<() => void>();

export function getServerStatus(): ServerStatus {
    if (slowRequests > 0 && !serverHasAnswered) return "waking";
    return unreachable ? "unreachable" : "ok";
}

export function subscribeServerStatus(listener: () => void): () => void {
    listeners.add(listener);
    return () => listeners.delete(listener);
}

export function dismissUnreachable() {
    unreachable = false;
    notify();
}

function notify() {
    listeners.forEach((listener) => listener());
}

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

export interface ServerRequestOptions {
    /** Retry on network errors and 502/503/504. Only for requests that are safe to repeat. */
    retry: boolean;
}

export async function serverFetch(url: string, init: RequestInit | undefined, options: ServerRequestOptions): Promise<Response> {
    let markedSlow = false;
    const slowTimer = setTimeout(() => {
        markedSlow = true;
        slowRequests++;
        notify();
    }, SLOW_AFTER_MS);

    const finish = (reachable: boolean) => {
        clearTimeout(slowTimer);
        if (markedSlow) slowRequests--;
        unreachable = !reachable;
        if (reachable) serverHasAnswered = true;
        notify();
    };

    if (!options.retry) {
        // One attempt only. Any HTTP response (including the backend's own 504 for a slow clone) goes back to the caller.
        try {
            const response = await fetch(url, init);
            finish(true);
            return response;
        } catch {
            finish(false);
            throw new Error(UNREACHABLE_MESSAGE);
        }
    }

    // Safe reads: retry network errors and 502/503/504 with backoff until the deadline.
    const controller = new AbortController();
    const deadline = setTimeout(() => controller.abort(), READ_DEADLINE_MS);
    try {
        for (let attempt = 0; !controller.signal.aborted; attempt++) {
            try {
                const response = await fetch(url, { ...init, signal: controller.signal });
                if (!WAKING_STATUSES.has(response.status)) {
                    finish(true);
                    return response;
                }
            } catch {
                // Network error or deadline abort; the loop condition decides.
            }
            await sleep(Math.min(1000 * 2 ** attempt, MAX_RETRY_DELAY_MS));
        }
    } finally {
        clearTimeout(deadline);
    }
    finish(false);
    throw new Error(UNREACHABLE_MESSAGE);
}

/** Error text for a failed response: the backend's `detail`, or a waking/unreachable hint for gateway errors. */
export async function responseError(response: Response, fallback: string): Promise<Error> {
    const body = await response.json().catch(() => ({}));
    if (typeof body?.detail === "string" && body.detail) return new Error(body.detail);
    if (WAKING_STATUSES.has(response.status)) return new Error(UNREACHABLE_MESSAGE);
    return new Error(fallback);
}
