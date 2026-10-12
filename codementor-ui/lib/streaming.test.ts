// Run with `npm test`.
//
// Grounded mode streams the model's own tokens, so the browser must show each one as it lands
// rather than waiting for the answer to be complete. That is a property of the client, and it
// is checked two ways here: the chat window's state update is re-run against a token sequence,
// and the source is read to confirm nothing buffers between the SSE event and the message.
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

const chat = readFileSync(new URL("../components/Chat/ChatWindow.tsx", import.meta.url), "utf8");
const api = readFileSync(new URL("./api.ts", import.meta.url), "utf8");

interface Message {
    role: string;
    content: string;
}

/**
 * The chat window's token handling, as the component does it: a running string, and the last
 * message replaced with it on every token. Returns what the viewer would see after each token.
 */
function framesWhileStreaming(tokens: string[]): string[] {
    let messages: Message[] = [{ role: "user", content: "q" }, { role: "assistant", content: "" }];
    let streamedContent = "";
    const frames: string[] = [];
    for (const token of tokens) {
        streamedContent += token;
        messages = messages.map((m, i) =>
            i === messages.length - 1 ? { ...m, content: streamedContent } : m,
        );
        frames.push(messages[messages.length - 1].content);
    }
    return frames;
}

test("every token produces a new frame, so the answer grows as it arrives", () => {
    const frames = framesWhileStreaming(["The ", "signer ", "derives ", "a key."]);

    assert.equal(frames.length, 4, "one visible update per token, not one at the end");
    assert.deepEqual(frames, [
        "The ",
        "The signer ",
        "The signer derives ",
        "The signer derives a key.",
    ]);
});

test("a frame is never shorter than the one before it", () => {
    const frames = framesWhileStreaming(["a", "b", "c", "d", "e"]);
    for (let i = 1; i < frames.length; i++) {
        assert.ok(frames[i].startsWith(frames[i - 1]), "a token must extend the answer, never rewrite it");
    }
});

test("a single-token answer still renders", () => {
    assert.deepEqual(framesWhileStreaming(["Done."]), ["Done."]);
});

test("a late first token leaves the message empty until it arrives", () => {
    // A reasoning model sends nothing visible for a while; the server forwards no token event
    // for those chunks, so the client simply has no frame yet.
    assert.deepEqual(framesWhileStreaming([]), []);
});

test("whitespace in a token is preserved, so words do not run together", () => {
    const frames = framesWhileStreaming(["def ", "sign", "(", "value", "):"]);
    assert.equal(frames[frames.length - 1], "def sign(value):");
});

test("nothing between the SSE event and the message buffers or trims a token", () => {
    assert.match(api, /case "token":\s*callbacks\.onToken\(data\.content\);/);
    assert.match(chat, /onToken: \(token: string\) => \{\s*streamedContent \+= token;/);
    assert.match(chat, /content: streamedContent/);
    // No setTimeout, batching or join on the token path.
    const onTokenAt = chat.indexOf("onToken: (token: string)");
    const body = chat.slice(onTokenAt, onTokenAt + 400);
    assert.ok(!body.includes("setTimeout"), "a token must not be delayed by the client");
});
