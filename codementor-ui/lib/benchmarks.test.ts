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
    headlineView,
    limitationsView,
    recordedLabel,
    retrievalView,
    structuralView,
    systemView,
    timingView,
} from "./benchmarks.ts";

/** The page source, for the few claims that are about layout rather than about a number. */
const SECTION = readFileSync(new URL("../components/BenchmarksSection.tsx", import.meta.url), "utf8");

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

    assert.deepEqual(overall.cells, ["0.700", "0.850", "0.850", "0.650", "0.778", "1.000"]);
});

test("the legacy hash numbers are drawn too, in their own clearly labelled tables", () => {
    const view = retrievalView(SUMMARY)!;
    const hash = view.providers.find((p) => p.id === "hash")!;
    const legacy = hash.splits.find((s) => s.id === "held-out (legacy)")!;
    const overall = legacy.rows.find((r) => r.scope === "overall")!;

    // hit@5 0.750 and MRR@10 0.645: the pair docs/RESULTS.md quotes in entry after entry.
    assert.deepEqual(overall.cells, ["0.600", "0.650", "0.750", "0.533", "0.645", "0.800"]);
    assert.match(legacy.note, /before the default change/);
    assert.match(legacy.note, /not the server's current numbers/);
});

test("every hash table says which configuration it was measured under", () => {
    const view = retrievalView(SUMMARY)!;
    const hash = view.providers.find((p) => p.id === "hash")!;

    assert.equal(hash.configuration, "structured records, BM25 fusion on");
    for (const split of hash.splits) {
        const expected = split.id.includes("legacy") ? "whole records, BM25 off" : "structured records, BM25 on";
        assert.equal(split.configuration, expected, split.id);
    }
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

test("the page says plainly that the original held-out split was reused", () => {
    const view = experimentsView(SUMMARY)!;

    assert.match(view.reusedSplitNote, /reused/i);
    assert.match(view.reusedSplitNote, /weak confirmation/);
    assert.match(view.freshSplitNote, /holdout2/);
    assert.match(view.freshSplitNote, /fresher/);
    assert.match(view.heldOutLabel, /reused/);
    assert.match(view.freshLabel, /scored once/);
});

test("the retrieval panel flags the reused split and leaves development alone", () => {
    const view = retrievalView(SUMMARY)!;

    assert.match(view.splitsNote, /reused/i);
    for (const provider of view.providers) {
        const heldOut = provider.splits.find((s) => s.id === "held-out")!;
        const development = provider.splits.find((s) => s.id === "development")!;
        assert.equal(heldOut.reused, true, `${provider.id}: the held-out split must be flagged as reused`);
        assert.match(heldOut.note, /reused/i);
        assert.equal(development.reused, false);
        assert.equal(development.note, "");
    }
});

test("both held-out sets are shown side by side, with the gap between them", () => {
    const compared = experimentsView(SUMMARY)!.splitsCompared!;

    assert.equal(compared.metric, "mrr@10");
    assert.equal(compared.rows.length, 4);
    for (const row of compared.rows) {
        assert.match(row.heldOut, /^\d\.\d{3}$/, `${row.key} must show a held-out number`);
        assert.match(row.holdout2, /^\d\.\d{3}$/, `${row.key} must show a fresh number`);
        assert.match(row.difference, /^[+-]\d\.\d{3}$/, `${row.key} must show the gap with its sign`);
        assert.ok(row.sources.includes("eval/results/"), `${row.key} must name where its numbers came from`);
    }
});

test("holdout3 is drawn as its own block, labelled as scored once", () => {
    const holdout3 = experimentsView(SUMMARY)!.holdout3!;

    assert.match(holdout3.label, /scored once/);
    assert.equal(holdout3.questions, 20);
    assert.deepEqual(holdout3.metrics, ["hit@1", "hit@3", "hit@5", "mrr@10", "recall@20"]);
    assert.match(holdout3.rule, /before holdout3 was scored/);
    // Passing the rule and being adopted are two different things, and the block says which
    // happened: hash became the default, cloudflare passed by 0.0071 of MRR@10 and did not.
    assert.match(holdout3.note, /only the hash one was adopted/);
    assert.match(holdout3.note, /cloudflare still runs whole records with BM25 off/);
    assert.ok(holdout3.recordedLabel.includes("recorded"));
    for (const row of holdout3.rows) {
        assert.equal(row.cells.length, holdout3.metrics.length, `${row.key} is missing a metric`);
        assert.ok(row.verdict.length > 0, `${row.key} has no verdict`);
        assert.ok(row.source.startsWith("eval/results/"), `${row.key} must name its result file`);
    }
});

test("the configuration the rule barred from selection is shown and marked ineligible", () => {
    // Leaving it out would hide where the hash gain comes from; drawing it like a candidate
    // would let a reader think it was one. It is drawn, and it says it is not eligible.
    const holdout3 = experimentsView(SUMMARY)!.holdout3!;
    const ineligible = holdout3.rows.filter((r) => !r.eligible);

    assert.equal(ineligible.length, 1);
    assert.match(ineligible[0].verdict, /not eligible/);
    assert.equal(ineligible[0].passes, false, "an ineligible row must never be marked as passing");
});

test("a holdout3 verdict is the server's, read rather than decided here", () => {
    const holdout3 = experimentsView(SUMMARY)!.holdout3!;

    for (const row of holdout3.rows) {
        assert.equal(row.passes, row.verdict.startsWith("passes"), `${row.key}`);
    }
    assert.equal(holdout3.rows.filter((r) => r.passes).length, 2, "one candidate per embedder passed");
});

test("the never-used split has a column of its own in both experiment tables", () => {
    const view = experimentsView(SUMMARY)!;

    assert.match(view.neverUsedLabel, /never used/);
    for (const row of view.rows) {
        if (row.verdict === "chosen" || row.verdict === "baseline") {
            assert.ok(row.holdout3, `${row.key} was scored on holdout3 and should show it`);
        } else {
            assert.equal(row.holdout3, null, `${row.key} must not claim a holdout3 score`);
        }
    }
    for (const row of view.splitsCompared!.rows) {
        assert.match(row.holdout3, /^\d\.\d{3}$/, `${row.key} must show its holdout3 number`);
        assert.ok(row.sources.includes("holdout3-"), `${row.key} must name the holdout3 file it read`);
    }
});

test("the page no longer tells a reader that one fresh set settles it", () => {
    // Two sets that were each scored once disagree with each other by more than either
    // disagrees with dev, which is a fact about twenty-question sets, not about one set.
    const view = experimentsView(SUMMARY)!;

    assert.match(view.freshSplitNote, /do not agree/);
    assert.match(view.bothSplitsNote, /All three/);
    assert.match(view.bothSplitsNote, /no single set/);
});

test("the side-by-side table keeps showing that the fresh split scores worse", () => {
    // The relationship, not the numbers: a question set that was never used scores lower than
    // one that was. If that ever stops being true the wording above has to change with it.
    for (const row of experimentsView(SUMMARY)!.splitsCompared!.rows) {
        assert.equal(row.worseOnFresh, true, `${row.key} should be worse on the fresh split`);
        assert.ok(Number(row.holdout2) < Number(row.heldOut), row.key);
        assert.ok(row.difference.startsWith("-"), row.key);
    }
});

test("a summary with no side-by-side block renders the rest of the panel", () => {
    const view = experimentsView({
        experiments: {
            metrics: ["mrr@10"],
            rows: [{ embedder: "hash", records: "whole", arrangement: "current", verdict: "baseline", dev: { questions: 20, "mrr@10": 0.5 } }],
            splits_compared: null,
        },
    })!;

    assert.equal(view.splitsCompared, null);
    assert.equal(view.rows.length, 1);
});

test("the chat-mode block says how large the run was, so a trial cannot pass for it", () => {
    const chat = experimentsView(SUMMARY)!.chatModes!;

    assert.equal(chat.questions, 15);
    assert.equal(chat.runs, 30);
    assert.match(chat.scaleNote, /15 questions, 30 runs/);
    assert.match(chat.scaleNote, /trial/);
    assert.equal(
        chat.rows.reduce((total, row) => total + Number(row.score.split("/")[1]), 0),
        chat.runs,
    );
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


/* ---- the headline block ---- */
//
// The first thing on the page, and for most readers the only thing. What it has to get right
// is that every figure in it comes from the committed summary, that the two configurations it
// names are the ones the server actually ships, and that it carries the three pieces of
// wording without which the numbers would be read as something they are not.

test("the headline shows both embedders at their shipped configuration", () => {
    const view = headlineView(SUMMARY)!;

    assert.deepEqual(
        view.rows.map((r) => r.key),
        ["hash", "cloudflare"],
    );
    assert.match(view.rows[0].configuration, /structured records, BM25 fusion on/);
    assert.match(view.rows[1].configuration, /whole records, no fusion/);
});

test("exactly one headline row is the default, and it is hash", () => {
    const view = headlineView(SUMMARY)!;
    const defaults = view.rows.filter((r) => r.isDefault);

    assert.equal(defaults.length, 1);
    assert.equal(defaults[0].key, "hash");
});

test("the headline reports MRR@10 and hit@5 on holdout2 and holdout3, with the counts", () => {
    const view = headlineView(SUMMARY)!;

    assert.deepEqual(view.metrics, ["mrr@10", "hit@5"]);
    for (const row of view.rows) {
        assert.deepEqual(
            row.splits.map((s) => s.id),
            ["holdout2", "holdout3"],
            row.key,
        );
        for (const split of row.splits) {
            assert.equal(split.questions, "20", `${row.key} ${split.id}`);
            assert.equal(split.cells.length, 2);
            for (const cell of split.cells) assert.match(cell, /^\d\.\d{3}$/, `${row.key} ${split.id}`);
        }
    }
});

test("the headline figures are the ones the Experiments table shows", () => {
    // The whole point of deriving the block from the built summary: a headline that disagrees
    // with the table it summarises is worse than no headline at all. Matched on is_default,
    // which is also the claim that the headline shows the row the server actually runs - for
    // hash that row is the chosen one and for cloudflare it is the baseline.
    const headline = headlineView(SUMMARY)!;

    for (const row of headline.rows) {
        const full = (SUMMARY.experiments?.rows ?? []).find((r) => r.embedder === row.key && r.is_default);
        assert.ok(full, `${row.key} has no default row in the experiments table`);
        for (const split of row.splits) {
            const scores = split.id === "holdout2" ? full!.holdout2 : full!.holdout3;
            assert.equal(split.cells[0], formatMetric(scores?.["mrr@10"]), `${row.key} ${split.id} mrr@10`);
            assert.equal(split.cells[1], formatMetric(scores?.["hit@5"]), `${row.key} ${split.id} hit@5`);
            assert.equal(split.questions, formatCount(scores?.questions), `${row.key} ${split.id} questions`);
        }
    }
});

test("the headline carries the structural precision and recall against ast", () => {
    const view = headlineView(SUMMARY)!;

    assert.match(view.structural!.title, /ast/);
    assert.equal(view.structural!.precision, "1.000");
    assert.equal(view.structural!.recall, "0.967");
    assert.equal(view.structural!.questions, 49);
});

test("the headline timing says the model is not in it", () => {
    const view = headlineView(SUMMARY)!;

    assert.equal(view.timing!.label, "excludes the language model");
    assert.equal(view.timing!.total, "45 ms");
    assert.equal(view.timing!.rerank, "19 ms");
    assert.match(view.timing!.note, /End-to-end latency is not shown/);
});

test("the headline repeats that these are recorded results, not live accuracy", () => {
    assert.equal(headlineView(SUMMARY)!.note, "Recorded results on small fixed question sets, not live accuracy.");
});

test("the headline says why these two sets and not the reused one", () => {
    assert.match(headlineView(SUMMARY)!.splitsNote, /scored once each/);
    assert.match(headlineView(SUMMARY)!.splitsNote, /reused/);
});

test("a metric a headline row never scored is a dash, not 0.000", () => {
    const view = headlineView({
        headline: {
            metrics: ["mrr@10", "hit@5"],
            retrieval: [{ embedder: "hash", holdout3: { questions: 20, "mrr@10": null } }],
        },
    })!;

    assert.deepEqual(view.rows[0].splits[0].cells, [NOT_MEASURED, NOT_MEASURED]);
});

test("a summary with no headline loses the block rather than breaking the page", () => {
    assert.equal(headlineView({}), null);
    assert.equal(headlineView({ headline: { retrieval: [] } }), null);
    assert.equal(benchmarksView({ available: false, heading_note: "x" }).headline, null);
});

test("a headline with no structural or timing data draws neither", () => {
    const view = headlineView({ headline: { retrieval: [{ embedder: "hash" }] } })!;

    assert.equal(view.structural, null);
    assert.equal(view.timing, null);
    assert.equal(view.rows[0].splits.length, 0);
});

test("the whole section has a headline when the summary does", () => {
    assert.notEqual(benchmarksView(RESPONSE).headline, null);
});

/* ---- everything else is behind one closed disclosure ---- */

test("the section renders the headline and nothing else at the top level", () => {
    // The six detail panels are collected into one list and handed to Details; only the
    // headline is drawn beside it.
    assert.match(SECTION, /\{view\.headline && <Headline view=\{view\.headline\} \/>\}/);
    assert.match(SECTION, /\{details\.length > 0 && <Details count=\{details\.length\}>\{details\}<\/Details>\}/);
});

test("every panel that is not the headline is inside the details list", () => {
    const list = SECTION.slice(SECTION.indexOf("const details = ["), SECTION.indexOf("].filter(Boolean)"));
    for (const panel of ["Retrieval", "Structural", "System", "Timing", "Experiments", "Limitations"]) {
        assert.ok(list.includes(`<${panel} key=`), `${panel} should be behind Details`);
    }
});

test("the disclosure is a native details element with no open attribute", () => {
    // Closed by default, closed on first paint, and opened by find-in-page without any
    // JavaScript. An `open` or a useState here would undo all three.
    const details = SECTION.slice(SECTION.indexOf("function Details("), SECTION.indexOf("/* ---- The section ---- */"));
    assert.match(details, /<details className=/);
    assert.ok(!details.includes("open="), "the panel must be closed by default");
    assert.ok(!details.includes("useState"), "no state: the browser owns whether it is open");
});
