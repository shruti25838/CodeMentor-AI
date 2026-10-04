// Run with `npm test`.
import { test } from "node:test";
import assert from "node:assert/strict";

import { parseCitation } from "./citations.ts";

test("function citation with a line range and snippet", () => {
    assert.deepEqual(parseCitation("src/itsdangerous/signer.py (lines 120-145) | def sign(self, value):"), {
        path: "src/itsdangerous/signer.py",
        startLine: 120,
        endLine: 145,
        snippet: "def sign(self, value):",
    });
});

test("file citation without a line range", () => {
    assert.deepEqual(parseCitation("README.md | # itsdangerous"), { path: "README.md", snippet: "# itsdangerous" });
});

test("bare path", () => {
    assert.deepEqual(parseCitation("setup.py"), { path: "setup.py", snippet: "" });
});

test("snippet containing the separator and parentheses stays in the snippet", () => {
    const parsed = parseCitation("a.py (lines 1-2) | x = f(1) | y (lines 3-4)");
    assert.equal(parsed.path, "a.py");
    assert.equal(parsed.startLine, 1);
    assert.equal(parsed.snippet, "x = f(1) | y (lines 3-4)");
});

test("reversed range is clamped to the start line", () => {
    const parsed = parseCitation("a.py (lines 9-3)");
    assert.equal(parsed.startLine, 9);
    assert.equal(parsed.endLine, 9);
});

test("path with spaces", () => {
    assert.equal(parseCitation("docs/my file.py (lines 1-1) | x").path, "docs/my file.py");
});
