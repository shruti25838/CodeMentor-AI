// Run with `npm test` (Node 22.6+ built-in test runner with type stripping; no extra dependencies).
import { afterEach, beforeEach, mock, test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

import {
    getServerStatus,
    READ_DEADLINE_MS,
    responseError,
    serverFetch,
    SLOW_AFTER_MS,
    UNREACHABLE_MESSAGE,
} from "./serverRequest.ts";

const realFetch = globalThis.fetch;
let calls = 0;

/** Each fetch call takes the next behavior: a Response, an Error to throw, or a delay before a Response. */
function fakeFetch(...behaviors: Array<Response | Error | { after: number; response: Response }>) {
    calls = 0;
    globalThis.fetch = (async (_url: unknown, init?: RequestInit) => {
        const behavior = behaviors[Math.min(calls++, behaviors.length - 1)];
        if (behavior instanceof Error) throw behavior;
        if (behavior instanceof Response) return behavior;
        return new Promise<Response>((resolve, reject) => {
            const timer = setTimeout(() => resolve(behavior.response), behavior.after);
            init?.signal?.addEventListener("abort", () => {
                clearTimeout(timer);
                reject(new Error("aborted"));
            });
        });
    }) as typeof fetch;
}

const flush = () => new Promise((resolve) => setImmediate(resolve));

/** Advance mocked time in 500 ms steps, letting pending promises run between steps. */
async function advance(ms: number) {
    for (let t = 0; t < ms; t += 500) {
        mock.timers.tick(500);
        await flush();
    }
}

beforeEach(() => mock.timers.enable({ apis: ["setTimeout"] }));
afterEach(() => {
    mock.timers.reset();
    globalThis.fetch = realFetch;
});

const ok = () => new Response("{}", { status: 200 });
const networkError = () => new TypeError("Failed to fetch");

test("read request retries network errors and recovers", async () => {
    fakeFetch(networkError(), networkError(), networkError(), ok());
    const pending = serverFetch("http://x/repos", undefined, { retry: true });
    await advance(SLOW_AFTER_MS);
    assert.equal(getServerStatus(), "waking");
    await advance(20_000);
    const response = await pending;
    assert.equal(response.status, 200);
    assert.equal(calls, 4);
    assert.equal(getServerStatus(), "ok");
});

test("read request retries gateway errors from a sleeping host", async () => {
    fakeFetch(new Response("", { status: 502 }), new Response("", { status: 503 }), ok());
    const pending = serverFetch("http://x/repos", undefined, { retry: true });
    await advance(10_000);
    assert.equal((await pending).status, 200);
    assert.equal(calls, 3);
});

test("read request gives up at the deadline with a clear message", async () => {
    fakeFetch(networkError());
    const pending = serverFetch("http://x/repos", undefined, { retry: true });
    const result = assert.rejects(pending, { message: UNREACHABLE_MESSAGE });
    await advance(READ_DEADLINE_MS + 10_000);
    await result;
    assert.equal(getServerStatus(), "unreachable");
});

test("request sent once is never retried on a network error", async () => {
    fakeFetch(networkError(), ok());
    const pending = serverFetch("http://x/analyze-repo", { method: "POST" }, { retry: false });
    const result = assert.rejects(pending, { message: UNREACHABLE_MESSAGE });
    await advance(30_000);
    await result;
    assert.equal(calls, 1);
    assert.equal(getServerStatus(), "unreachable");
});

test("request sent once is never retried on a gateway error and gets the response as-is", async () => {
    fakeFetch(new Response('{"detail":"The clone took too long."}', { status: 504 }), ok());
    const response = await serverFetch("http://x/analyze-repo", { method: "POST" }, { retry: false });
    await advance(30_000);
    assert.equal(response.status, 504);
    assert.equal(calls, 1);
});

test("slow request sent once shows waking, then clears when it answers", async () => {
    fakeFetch({ after: 40_000, response: ok() });
    const pending = serverFetch("http://x/analyze-repo", { method: "POST" }, { retry: false });
    await advance(SLOW_AFTER_MS - 500);
    assert.equal(getServerStatus(), "ok");
    await advance(1000);
    assert.equal(getServerStatus(), "waking");
    await advance(40_000);
    assert.equal((await pending).status, 200);
    assert.equal(calls, 1);
    assert.equal(getServerStatus(), "ok");
});

test("responseError prefers the backend detail, then explains gateway errors", async () => {
    const withDetail = await responseError(new Response('{"detail":"Repository not found"}', { status: 404 }), "x");
    assert.equal(withDetail.message, "Repository not found");
    const gateway = await responseError(new Response("<html>Bad Gateway</html>", { status: 502 }), "x");
    assert.equal(gateway.message, UNREACHABLE_MESSAGE);
    const other = await responseError(new Response("", { status: 500 }), "Fallback");
    assert.equal(other.message, "Fallback");
});

test("api.ts sends indexing and chat once, and retries only read requests", () => {
    const source = readFileSync(new URL("./api.ts", import.meta.url), "utf8");
    const modes: Record<string, string> = {};
    for (const match of source.matchAll(/serverFetch\(`\$\{BASE_URL\}(\/[\w\-/]*)`[\s\S]*?(READ|ONCE)\)/g)) {
        modes[match[1]] = match[2];
    }
    assert.deepEqual(modes, {
        "/ask": "ONCE",
        "/ask/stream": "ONCE",
        "/analyze-repo": "ONCE",
        "/files": "READ",
        "/files/content": "READ",
        "/repo-overview": "READ",
        "/repos": "READ",
        "/dependencies/graph": "READ",
        "/eval/stats": "READ",
    });
    assert.ok(!/\bfetch\(/.test(source), "every request goes through serverFetch");
});
