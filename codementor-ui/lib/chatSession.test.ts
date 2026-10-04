// Run with `npm test` (Node 22.6+ built-in test runner with type stripping; no extra dependencies).
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

import { newChatSessionId } from "./chatSession.ts";

const SERVER_PATTERN = /^[A-Za-z0-9_-]{16,128}$/;

test("chat session ids are random and accepted by the server", () => {
    const ids = new Set(Array.from({ length: 100 }, () => newChatSessionId()));
    assert.equal(ids.size, 100);
    for (const id of ids) assert.match(id, SERVER_PATTERN);
});

test("falls back to getRandomValues outside a secure context", () => {
    const id = newChatSessionId({ getRandomValues: globalThis.crypto.getRandomValues.bind(globalThis.crypto) });
    assert.match(id, /^[0-9a-f]{32}$/);
});

test("the streamed chat sends the session id, and the chat window keeps one per conversation", () => {
    const api = readFileSync(new URL("./api.ts", import.meta.url), "utf8");
    assert.match(api, /JSON\.stringify\(\{ question, repo_id: repoId, session_id: sessionId \}\)/);
    const chat = readFileSync(new URL("../components/Chat/ChatWindow.tsx", import.meta.url), "utf8");
    assert.match(chat, /useState\(\(\) => newChatSessionId\(\)\)/);
    assert.ok(!/localStorage[^\n]*session|sessionStorage/.test(chat), "the id is not stored in the browser");
});
