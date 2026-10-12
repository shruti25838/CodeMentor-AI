// Run with `npm test`.
//
// These cover what the Benchmarks section ends up showing: the rows and labels it draws from a
// real response, what it does when data is missing, and the wording it is required to carry.
// The page maps over these functions and invents nothing, so testing them tests the section.
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

// Types come in on their own line: node --test strips types but cannot erase a type name
// imported alongside values.
import type { BenchmarksResponse, Summary } from "./benchmarks.ts";
import {
    NOT_MEASURED,
    benchmarksView,
    defaultProviderId,
    experimentsView,
    formatCount,
    formatMetric,
    formatShare,
    limitationsView,
    recordedLabel,
    retrievalView,
    structuralView,
    systemView,
    timingView,
} from "./benchmarks.ts";

/** The committed summary itself, so these tests move when the real data moves. */
const SUMMARY: Summary = JSON.parse(readFileSync(new URL("../../eval/benchmarks.json", import.meta.url), "utf8"));

const RESPONSE: BenchmarksResponse = {
    available: true,
    heading_note: SUMMARY.heading_note,
    live: { embedding_provider: "hash", semantic_document_cap: 600 },
    summary: SUMMARY,
};

/* ---- formatting ---- */

test("a metric that was never computed is a dash, not 0.000", () => {
    assert.equal(formatMetric(null), NOT_MEASURED);
    assert.equal(formatMetric(undefined), NOT_MEASURED);
    assert.equal(formatMetric("0.75"), NOT_MEASURED);
    assert.equal(formatMetric(NaN), NOT_MEASURED);
    assert.equal(formatMetric(0), "0.000");
    assert.equal(formatMetric(0.75), "0.750");
});

test("counts and shares have their own formats, and a missing one is a dash", () => {
    assert.equal(formatCount(1705), "1,705");
    assert.equal(formatCount(0), "0");
    assert.equal(formatCount(null), NOT_MEASURED);
    assert.equal(formatShare(0.241), "24.1%");
    assert.equal(formatShare(undefined), NOT_MEASURED);
});

test("every panel label names the date and the commit it was recorded at", () => {
    assert.equal(recordedLabel({ date: "2026-10-11", commit: "016c655" }), "recorded 2026-10-11 · 016c655");
    assert.equal(recordedLabel(undefined), "");
    assert.equal(recordedLabel({}), "");
});

/* ---- retrieval ---- */

test("retrieval renders a row per repository and one for the overall score", () => {
    const view = retrievalView(SUMMARY)!;
    const hash = view.providers.find((p) => p.id === "hash")!;
    const heldOut = hash.splits.find((s) => s.id === "held-out")!;

    assert.deepEqual(
        heldOut.rows.map((r) => r.scope),
        ["itsdangerous", "flask", "overall"],
    );
    assert.equal(heldOut.questions, 20);
    assert.equal(heldOut.rows.every((r) => r.cells.length === view.metrics.length), true);
    assert.deepEqual(view.metrics, ["hit@1", "hit@3", "hit@5", "p@3", "mrr@10", "recall@20"]);
});

test("the held-out hash numbers are the ones in the committed summary", () => {
    const view = retrievalView(SUMMARY)!;
    const hash = view.providers.find((p) => p.id === "hash")!;
    const overall = hash.splits.find((s) => s.id === "held-out")!.rows.find((r) => r.scope === "overall")!;

    assert.deepEqual(overall.cells, ["0.600", "0.650", "0.750", "0.533", "0.645", "0.800"]);
});

test("hash is the default panel and cloudflare is the opt-in one", () => {
    const view = retrievalView(SUMMARY)!;

    assert.equal(defaultProviderId(view), "hash");
    assert.equal(view.providers.find((p) => p.id === "cloudflare")!.isDefault, false);
    assert.match(view.providers.find((p) => p.id === "cloudflare")!.label, /opt-in/);
});

test("every retrieval split carries its embedder, question count, commit and date", () => {
    const view = retrievalView(SUMMARY)!;

    for (const provider of view.providers) {
        assert.ok(provider.embedder.length > 0, `${provider.id} has no embedder`);
        for (const split of provider.splits) {
            assert.equal(split.questions, 20);
            assert.match(split.recordedLabel, /^recorded \d{4}-\d{2}-\d{2} · [0-9a-f]{7}/);
        }
    }
});

test("cloudflare's development P@3 shows as a dash, because that run did not compute it", () => {
    const view = retrievalView(SUMMARY)!;
    const cloudflare = view.providers.find((p) => p.id === "cloudflare")!;
    const p3 = view.metrics.indexOf("p@3");

    const dev = cloudflare.splits.find((s) => s.id === "development")!;
    assert.equal(dev.rows.every((r) => r.cells[p3] === NOT_MEASURED), true);
    const heldOut = cloudflare.splits.find((s) => s.id === "held-out")!;
    assert.equal(heldOut.rows.every((r) => r.cells[p3] !== NOT_MEASURED), true);
});

test("retrieval with no providers renders nothing rather than an empty table", () => {
    assert.equal(retrievalView({ retrieval: { providers: [] } }), null);
    assert.equal(retrievalView({}), null);
    assert.equal(defaultProviderId(null), "");
});

/* ---- structural ---- */

test("structural renders the three groupings and every disagreement", () => {
    const view = structuralView(SUMMARY)!;

    assert.equal(view.language, "python");
    assert.equal(view.questions, 49);
    assert.deepEqual(
        view.groups.map((g) => g.heading),
        ["by split", "by repository", "by question kind"],
    );
    assert.equal(view.overall.precision, "1.000");
    assert.equal(view.overall.exact, "45/49");
    assert.equal(view.disagreements.length, 4);
    assert.match(view.disagreements[0].title, /who_calls/);
    assert.ok(view.disagreements[0].missed.length > 0);
    assert.equal(view.disagreements.every((d) => d.unexpected.length === 0), true);
});

test("a structural group with no questions shows a dash instead of a score", () => {
    const view = structuralView({
        structural: {
            overall: { questions: 0, precision: null, recall: null, exact: 0 },
            by_kind: { who_imports: { questions: 0, precision: null, recall: null, exact: 0 } },
        },
    })!;

    assert.equal(view.overall.precision, NOT_MEASURED);
    assert.equal(view.overall.exact, NOT_MEASURED);
    assert.equal(view.groups[0].rows[0].recall, NOT_MEASURED);
});

test("structural with no data renders nothing", () => {
    assert.equal(structuralView({}), null);
    assert.equal(structuralView({ structural: {} }), null);
});

/* ---- system facts ---- */

test("system facts show the routes, the live provider and the cap, and the call graph", () => {
    const view = systemView(SUMMARY, RESPONSE.live)!;

    assert.deepEqual(
        view.endpoints.map((f) => f.label),
        ["Routers", "Routes", "Open", "Need a key", "Rate limited"],
    );
    assert.equal(view.endpoints.find((f) => f.label === "Routes")!.value, String(SUMMARY.system.endpoints.routes));
    assert.deepEqual(
        view.live.map((f) => f.value),
        ["hash", "600"],
    );
    assert.deepEqual(
        view.callGraph.map((r) => r.repo),
        ["itsdangerous", "flask"],
    );
    assert.equal(view.callGraph[0].facts.find((f) => f.label === "Resolved")!.value, "91 (24.1%)");
});

test("system facts fall back to a dash when the server sends no live block", () => {
    const view = systemView(SUMMARY, undefined)!;

    assert.deepEqual(
        view.live.map((f) => f.value),
        [NOT_MEASURED, NOT_MEASURED],
    );
    // The labels and the hints are still there, so the panel explains what is missing.
    assert.match(view.live[0].hint!, /next index/);
});

/* ---- timing ---- */

test("the timing panel is labelled as excluding the language model", () => {
    const view = timingView(SUMMARY)!;

    assert.equal(view.label, "excludes the language model");
    assert.match(view.endToEndNote, /depends on the model provider and is not benchmarked here/);
    assert.equal(view.stages!.columns.includes("llm_stand_in"), true);
    assert.equal(view.stages!.rows.length, 4);
    assert.deepEqual(view.stages!.rows[1], { case: "itsdangerous warm", values: ["0.3", "0.4", "4.7", "3.1", "12.2", "25.5"] });
    assert.match(view.current!.note, /45 ms/);
});

test("the timing panel shows no end-to-end latency figure", () => {
    const view = timingView(SUMMARY)!;
    const printed = JSON.stringify(view).toLowerCase();

    for (const banned of ["ttfb", "end_to_end", "p95"]) {
        assert.equal(printed.includes(banned), false, `the timing panel must not show ${banned}`);
    }
});

test("timing renders what it has when a half is missing", () => {
    const view = timingView({ timing: { label: "excludes the language model" } })!;

    assert.equal(view.current, null);
    assert.equal(view.stages, null);
    assert.equal(view.label, "excludes the language model");
    assert.equal(timingView({}), null);
});

/* ---- limitations ---- */

test("the limitations panel is collapsed and carries the smoke check, samples, repos and cap", () => {
    const view = limitationsView(SUMMARY)!;

    assert.equal(view.collapsed, true);
    assert.equal(view.smoke!.modes.length, 2);
    assert.deepEqual(
        view.smoke!.modes.map((m) => `${m.mode} ${m.score}`),
        ["fast 8/15", "deep 3/15"],
    );
    assert.match(view.smoke!.whatItIs, /no judge model/);
    assert.deepEqual(
        view.sampleSizes.map((s) => s.questions),
        [40, 49, 15],
    );
    assert.deepEqual(
        view.repos.map((r) => r.name),
        ["itsdangerous", "flask"],
    );
    assert.equal(view.cap!.value, "600");
    assert.match(view.cap!.note, /not been measured/);
});

test("limitations renders with no smoke check and no cap", () => {
    const view = limitationsView({ limitations: { title: "Limitations", collapsed: false } })!;

    assert.equal(view.smoke, null);
    assert.equal(view.cap, null);
    assert.equal(view.collapsed, false);
    assert.deepEqual(view.sampleSizes, []);
    assert.equal(limitationsView({}), null);
});

/* ---- the section as a whole ---- */

test("the whole section renders from a real response", () => {
    const view = benchmarksView(RESPONSE);

    assert.equal(view.available, true);
    assert.equal(view.unavailableNote, "");
    assert.match(view.headingNote, /not live accuracy/);
    for (const panel of [view.retrieval, view.structural, view.system, view.timing, view.limitations]) {
        assert.notEqual(panel, null);
    }
});

test("a server with no summary file still renders a heading note and an explanation", () => {
    const view = benchmarksView({
        available: false,
        heading_note: "Recorded results on small fixed question sets, not live accuracy.",
        note: "Recorded results are not available in this build.",
        live: { embedding_provider: "hash", semantic_document_cap: 600 },
    });

    assert.equal(view.available, false);
    assert.match(view.headingNote, /not live accuracy/);
    assert.match(view.unavailableNote, /not available in this build/);
    for (const panel of [view.retrieval, view.structural, view.system, view.timing, view.limitations]) {
        assert.equal(panel, null);
    }
});

test("no response at all still renders the heading note", () => {
    for (const response of [undefined, null, {}]) {
        const view = benchmarksView(response);
        assert.equal(view.available, false);
        assert.match(view.headingNote, /not live accuracy/);
        assert.ok(view.unavailableNote.length > 0);
    }
});

test("a response whose summary is missing a block drops that panel and keeps the rest", () => {
    const partial: Summary = { ...SUMMARY };
    delete partial.structural;
    delete partial.timing;

    const view = benchmarksView({ ...RESPONSE, summary: partial });

    assert.equal(view.available, true);
    assert.equal(view.structural, null);
    assert.equal(view.timing, null);
    assert.notEqual(view.retrieval, null);
    assert.notEqual(view.limitations, null);
});

test("nothing in the section carries a question, a key or a spend figure", () => {
    const printed = JSON.stringify(benchmarksView(RESPONSE)).toLowerCase();

    for (const banned of ["api_key", "account_id", "neuron", "budget", "secret"]) {
        assert.equal(printed.includes(banned), false, `the section must not carry ${banned}`);
    }
});

/* ---- experiments ---- */

test("every experiment is listed, kept and rejected, with a verdict", () => {
    const view = experimentsView(SUMMARY)!;

    assert.equal(view.rows.length, 20, "four dev runs of five arrangements each");
    assert.ok(view.keepRule.includes("MRR@10 rises"));
    assert.deepEqual(view.metrics, ["hit@5", "mrr@10", "recall@20"]);
    for (const row of view.rows) {
        assert.ok(row.verdict.length > 0, `${row.key} has no verdict`);
        assert.equal(row.dev.length, view.metrics.length);
    }
    assert.ok(view.rows.some((r) => r.verdict === "rejected"), "rejected rows are shown too");
});

test("exactly one configuration per embedder is marked chosen", () => {
    const view = experimentsView(SUMMARY)!;
    const chosen = view.rows.filter((r) => r.verdict === "chosen");

    assert.deepEqual(
        chosen.map((r) => `${r.embedder}:${r.records}/${r.arrangement}`).sort(),
        ["cloudflare:structured/current", "hash:structured/rrf+keyword"],
    );
});

test("only the chosen and the baseline rows carry held-out or fresh numbers", () => {
    const view = experimentsView(SUMMARY)!;

    for (const row of view.rows) {
        if (row.verdict === "chosen") {
            assert.ok(row.heldOut, `${row.key} is chosen and should have held-out scores`);
            assert.ok(row.holdout2, `${row.key} is chosen and should have fresh scores`);
        } else if (row.verdict === "baseline") {
            assert.ok(row.holdout2, "the baseline is scored on the fresh split for comparison");
        } else {
            assert.equal(row.heldOut, null, `${row.key} must not claim a held-out score`);
            assert.equal(row.holdout2, null, `${row.key} must not claim a fresh score`);
        }
    }
});

test("the fresh split scores worse than dev for both baselines, and the table shows it", () => {
    const view = experimentsView(SUMMARY)!;
    const baselines = view.rows.filter((r) => r.verdict === "baseline");

    assert.equal(baselines.length, 2);
    for (const row of baselines) {
        const dev = Number(row.dev[1]);
        const fresh = Number(row.holdout2![1]);
        assert.ok(fresh < dev, `${row.embedder}: the fresh split should be the harder one`);
    }
});

test("an arrangement the server cannot be set to says so in its verdict", () => {
    const view = experimentsView(SUMMARY)!;
    const bm25 = view.rows.find((r) => r.embedder === "hash" && r.records === "whole" && r.arrangement === "bm25")!;

    assert.match(bm25.verdict, /no mode for it/);
});

test("the chat mode comparison is labelled as one local run on one model", () => {
    const view = experimentsView(SUMMARY)!;
    const chat = view.chatModes!;

    assert.match(chat.label, /one local run/);
    assert.match(chat.label, /not a benchmark/);
    assert.ok(chat.model.length > 0);
    assert.ok(chat.weakCheck.includes("No judge model"));
    assert.deepEqual(chat.rows.map((r) => r.mode), ["legacy", "grounded"]);
    for (const row of chat.rows) {
        assert.match(row.firstToken, /^(\d+\.\d\ds|—)$/);
    }
});

test("the experiments panel shows no end-to-end latency claim", () => {
    const printed = JSON.stringify(experimentsView(SUMMARY)).toLowerCase();
    for (const banned of ["ttfb", "end_to_end", "p95", "latency"]) {
        assert.equal(printed.includes(banned), false, `the experiments panel must not claim ${banned}`);
    }
});

test("experiments with no rows renders nothing rather than an empty table", () => {
    assert.equal(experimentsView({ experiments: { rows: [] } }), null);
    assert.equal(experimentsView({}), null);
});

test("a summary with experiments but no chat run still renders the table", () => {
    const view = experimentsView({
        experiments: {
            metrics: ["mrr@10"],
            rows: [{ embedder: "hash", records: "whole", arrangement: "current", verdict: "baseline", dev: { questions: 20, "mrr@10": 0.5 } }],
            chat_modes: null,
        },
    })!;

    assert.equal(view.chatModes, null);
    assert.equal(view.rows[0].dev[0], "0.500");
});

test("the whole section includes experiments when the summary has them", () => {
    assert.notEqual(benchmarksView(RESPONSE).experiments, null);
    assert.equal(benchmarksView({ available: false, heading_note: "x" }).experiments, null);
});

test("every experiment row carries the number of questions it was scored on", () => {
    const view = experimentsView(SUMMARY)!;
    for (const row of view.rows) {
        assert.equal(row.questions, 20, `${row.key} should say how many questions it scored`);
    }
});
