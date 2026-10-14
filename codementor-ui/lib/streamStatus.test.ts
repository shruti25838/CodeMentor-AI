// Run with `npm test`.
//
// The waiting indicator: what it says, which files it names, and when it goes away. The
// server sends a status event the moment retrieval finishes and before the model is asked
// anything, so the page can say "Found 3 files" and name them while the model is still
// thinking. Two things have to hold, and both are properties of the client:
//
//   - it only ever names files the server actually sent, because the server only sends them
//     where it can promise they are the citations the answer will carry;
//   - it clears the moment the first token lands, because from then on the answer is on
//     screen and a spinner over it would be a lie.
//
// The state machine is re-run here against event sequences, and the component source is read
// to confirm it uses these functions rather than wording the same thing again.
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

import {
    fileSummary,
    hasFoundFiles,
    isStatusVisible,
    STAGE_RETRIEVED,
    statusFromEvent,
    type ChatStatus,
    type StreamStatus,
} from "./streamStatus.ts";

const chat = readFileSync(new URL("../components/Chat/ChatWindow.tsx", import.meta.url), "utf8");

const RETRIEVED: StreamStatus = {
    stage: STAGE_RETRIEVED,
    content: "Found 2 files",
    count: 2,
    files: [
        { path: "src/itsdangerous/signer.py", start_line: 10, end_line: 42 },
        { path: "src/itsdangerous/timed.py", start_line: 1, end_line: 30 },
    ],
};

/** The component's own sequence: status events, then tokens, then done. */
function replay(events: ({ status: StreamStatus } | { token: string } | { done: true })[]) {
    let status: ChatStatus | null = null;
    let answer = "";
    const frames: { visible: boolean; label: string; files: string[] }[] = [];
    for (const event of events) {
        if ("status" in event) status = statusFromEvent(event.status);
        if ("token" in event) answer += event.token;
        if ("done" in event) status = null;
        frames.push({
            visible: isStatusVisible(status, answer),
            label: status?.label ?? "",
            files: status?.files ?? [],
        });
    }
    return frames;
}

test("the retrieved event renders as a count and the file names", () => {
    const status = statusFromEvent(RETRIEVED);

    assert.equal(status.stage, STAGE_RETRIEVED);
    assert.equal(status.label, "Found 2 files");
    assert.deepEqual(status.files, ["src/itsdangerous/signer.py", "src/itsdangerous/timed.py"]);
    assert.equal(hasFoundFiles(status), true);
    assert.equal(fileSummary(status), "signer.py, timed.py");
});

test("the tooltip keeps the line range of every location", () => {
    assert.deepEqual(statusFromEvent(RETRIEVED).locations, [
        "src/itsdangerous/signer.py (lines 10-42)",
        "src/itsdangerous/timed.py (lines 1-30)",
    ]);
});

test("two windows of one file are named once", () => {
    const status = statusFromEvent({
        stage: STAGE_RETRIEVED,
        content: "Found 1 file",
        count: 1,
        files: [
            { path: "src/flask/app.py", start_line: 1, end_line: 40 },
            { path: "src/flask/app.py", start_line: 300, end_line: 340 },
        ],
    });

    assert.deepEqual(status.files, ["src/flask/app.py"]);
    assert.equal(status.locations.length, 2, "the tooltip still shows both ranges");
    assert.equal(fileSummary(status), "app.py");
});

test("a long list is cut short rather than filling the pane", () => {
    const paths = ["a.py", "b.py", "c.py", "d.py", "e.py", "f.py"];
    const status = statusFromEvent({
        stage: STAGE_RETRIEVED,
        content: "Found 6 files",
        files: paths.map((path) => ({ path, start_line: 1, end_line: 2 })),
    });

    assert.equal(fileSummary(status), "a.py, b.py, c.py, d.py and 2 more");
});

test("a stage-only event shows its sentence and names no files", () => {
    // Every path that cannot promise the citations sends one of these, and the client must
    // not invent a file list from it.
    const status = statusFromEvent({ stage: "retrieving", content: "Retrieving context..." });

    assert.equal(status.label, "Retrieving context...");
    assert.deepEqual(status.files, []);
    assert.equal(hasFoundFiles(status), false);
    assert.equal(fileSummary(status), "");
});

test("an older server that sends no stage still renders its sentence", () => {
    const status = statusFromEvent({ content: "Thinking..." } as StreamStatus);

    assert.equal(status.label, "Thinking...");
    assert.equal(status.stage, "");
    assert.equal(hasFoundFiles(status), false);
});

test("the status renders while waiting and clears on the first token", () => {
    const frames = replay([
        { status: { stage: "retrieving", content: "Retrieving context..." } },
        { status: RETRIEVED },
        { token: "It " },
        { token: "signs." },
        { done: true },
    ]);

    assert.deepEqual(
        frames.map((f) => f.visible),
        [true, true, false, false, false],
    );
    assert.equal(frames[1].label, "Found 2 files");
    assert.deepEqual(frames[1].files, ["src/itsdangerous/signer.py", "src/itsdangerous/timed.py"]);
});

test("done clears the status even when no token ever arrived", () => {
    const frames = replay([{ status: RETRIEVED }, { done: true }]);

    assert.equal(frames[0].visible, true);
    assert.equal(frames[1].visible, false);
});

test("nothing is shown before the first status event", () => {
    assert.equal(isStatusVisible(null, ""), false);
    assert.equal(isStatusVisible(null, "half an answer"), false);
});

test("the chat window uses these functions rather than wording it again", () => {
    assert.match(chat, /statusFromEvent\(status\)/);
    assert.match(chat, /hasFoundFiles\(streamStatus\)/);
    assert.match(chat, /isStatusVisible\(streamStatus, ""\)/);
    assert.match(chat, /fileSummary\(streamStatus\)/);
    // The status is dropped on done and on error, not only when a token arrives.
    assert.match(chat, /onDone: \(data\) => \{\s*setStreamStatus\(null\);/);
    assert.match(chat, /onError: \(error: string\) => \{\s*setStreamStatus\(null\);/);
});
