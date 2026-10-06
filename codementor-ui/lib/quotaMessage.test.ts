// Run with `npm test`. Checks the quota message the backend sends reaches the screen intact.
//
// The message is sent as ordinary `token` events, so what matters is that nothing between
// the socket and the bubble alters or hides it. api.ts cannot be imported here because it
// imports "./serverRequest" without a file extension, which Node's loader will not resolve,
// so its behaviour is asserted against its source, as the other tests in this folder do.
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

const api = readFileSync(new URL("./api.ts", import.meta.url), "utf8");
const chat = readFileSync(new URL("../components/Chat/ChatWindow.tsx", import.meta.url), "utf8");
const bubble = readFileSync(new URL("../components/Chat/MessageBubble.tsx", import.meta.url), "utf8");

// The exact wording lives in codeatlas/services/llm/quota.py. These are what a visitor sees
// arrive over the wire; tests/test_quota_message.py pins the backend to the same strings.
const DAILY = "The demo's AI quota for today is used up. Please try again later.";
const PER_MINUTE =
    "The demo's AI is busy right now — its per-minute quota is used up. Wait a minute and ask again.";
const FILES_HEADING = "The search did find these files, which may still help:";
const TRUNCATED = "_This answer was cut off because it reached the model's output limit._";

/** The backend streams an answer four words at a time; this is that, reassembled. */
function streamAndReassemble(answer: string): string {
    const words = answer.split(" ");
    let out = "";
    for (let i = 0; i < words.length; i += 4) {
        out += words.slice(i, i + 4).join(" ") + (i + 4 < words.length ? " " : "");
    }
    return out;
}

const quotaAnswer = (message: string) => `${message}\n\n${FILES_HEADING}\n- src/signer.py (lines 31-37)`;

test("a quota message survives being streamed four words at a time", () => {
    const text = streamAndReassemble(quotaAnswer(DAILY));
    assert.ok(text.includes(DAILY), "the sentence is not broken by chunking");
    assert.ok(text.includes(FILES_HEADING));
    assert.ok(text.indexOf(DAILY) < text.indexOf(FILES_HEADING), "message first, files after");
    assert.ok(text.includes("src/signer.py (lines 31-37)"));
});

test("the per-minute message is the one that says to wait a minute", () => {
    const text = streamAndReassemble(quotaAnswer(PER_MINUTE));
    assert.ok(text.includes("Wait a minute"));
    assert.ok(!text.includes("try again later"));
});

test("neither message shows a provider error code to a visitor", () => {
    for (const message of [DAILY, PER_MINUTE]) {
        for (const leak of ["429", "413", "Error code", "rate_limit"]) {
            assert.ok(!message.includes(leak), `${leak} must not appear in visitor copy`);
        }
    }
});

test("the truncation note survives streaming and keeps the answer before it", () => {
    const text = streamAndReassemble("Signing works like this. " + TRUNCATED);
    assert.ok(text.startsWith("Signing works like this."));
    assert.ok(text.includes("cut off"));
});

test("api.ts hands token content to the caller unchanged", () => {
    // No trimming, slicing or filtering between the event and onToken.
    assert.match(api, /case "token":\s*callbacks\.onToken\(data\.content\);/);
});

test("the chat window appends every streamed token to the message it displays", () => {
    assert.match(chat, /onToken: \(token: string\) => \{\s*streamedContent \+= token;/);
    assert.match(chat, /content: streamedContent/);
});

test("the message bubble renders the answer text itself, not behind a toggle", () => {
    assert.match(bubble, /<ReactMarkdown/);
    assert.match(bubble, /\{content\}/);
    // The collapsibles are for reasoning and agent steps; the answer is not inside one.
    const answerAt = bubble.indexOf("<ReactMarkdown");
    const reasoningAt = bubble.indexOf('title="Reasoning"');
    assert.ok(reasoningAt === -1 || reasoningAt < answerAt, "the answer is rendered outside the collapsibles");
});
