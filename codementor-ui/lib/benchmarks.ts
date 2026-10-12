/**
 * The Benchmarks section's view model: everything GET /eval/benchmarks sends, turned into the
 * rows and labels the page draws, with nothing drawn for data that is not there.
 *
 * It lives here rather than in the page so that it can be tested. The page maps over what these
 * functions return and adds no numbers and no wording of its own, so a test on these covers what
 * a reader ends up seeing.
 *
 * Two rules the whole section depends on:
 *   - these are recorded results, not live accuracy, and every panel says when it was recorded;
 *   - a metric a run did not compute shows as a dash, never as 0.000.
 *
 * Every field below is optional, because the summary is a file on the server: an older or a
 * partial one must lose a panel, not break the page.
 */

/* ---- what the server sends ---- */

export interface Recorded {
    commit?: string;
    date?: string;
    results_entry?: string;
}

export interface MetricRow {
    scope: string;
    questions: number;
    [metric: string]: string | number | null | undefined;
}

export interface Split {
    id: string;
    questions: number;
    source?: string;
    recorded?: Recorded;
    rows?: MetricRow[];
}

export interface Provider {
    id: string;
    label: string;
    default?: boolean;
    embedder?: string;
    note?: string;
    splits?: Split[];
}

export interface RetrievalBlock {
    title?: string;
    scored?: string;
    question_set?: string;
    metrics?: string[];
    providers?: Provider[];
}

export interface Group {
    questions?: number;
    precision?: number | null;
    recall?: number | null;
    exact?: number;
}

export interface Disagreement {
    repo?: string;
    kind?: string;
    subject?: string;
    note?: string;
    ast_found_graph_missed?: string[];
    graph_found_ast_did_not?: string[];
}

export interface StructuralBlock {
    title?: string;
    note?: string;
    language?: string;
    recorded?: Recorded;
    overall?: Group;
    by_split?: Record<string, Group>;
    by_repo?: Record<string, Group>;
    by_kind?: Record<string, Group>;
    disagreements?: Disagreement[];
    disagreements_note?: string;
}

export interface CallGraphRepo {
    modules?: number;
    definitions?: number;
    call_sites?: number;
    resolved?: number;
    resolved_share?: number;
    import_cycle_groups?: number;
}

export interface SystemBlock {
    title?: string;
    endpoints?: {
        routers?: number;
        routes?: number;
        open?: number;
        need_a_key?: number;
        rate_limited?: number;
        recorded?: Recorded;
    };
    call_graph?: { recorded?: Recorded; note?: string; repos?: Record<string, CallGraphRepo> };
}

export interface TimingBlock {
    title?: string;
    label?: string;
    units?: string;
    end_to_end_note?: string;
    current?: { note?: string; recorded?: Recorded };
    stages?: {
        note?: string;
        recorded?: Recorded;
        columns?: string[];
        rows?: { case: string; values?: number[] }[];
    };
}

export interface LimitationsBlock {
    title?: string;
    collapsed?: boolean;
    smoke_check?: {
        title?: string;
        what_it_is?: string;
        model?: string;
        note?: string;
        modes?: {
            mode: string;
            passed: number;
            of: number;
            expected_file_cited: number;
            fact_present: number;
            recorded?: Recorded;
        }[];
    };
    sample_sizes?: { what: string; questions: number; detail: string }[];
    sample_size_note?: string;
    repos?: { name: string; url: string; commit: string }[];
    repos_note?: string;
    document_cap?: { documents?: number; applies_to?: string; note?: string; recorded?: Recorded };
}

export interface ExperimentScores {
    questions?: number;
    [metric: string]: number | null | undefined;
}

export interface ExperimentRow {
    embedder?: string;
    records?: string;
    arrangement?: string;
    change?: string;
    verdict?: string;
    dev?: ExperimentScores | null;
    held_out?: ExperimentScores | null;
    holdout2?: ExperimentScores | null;
    recorded?: Recorded;
}

export interface ChatModeRow {
    mode: string;
    runs: number;
    passed: number;
    expected_file_cited: number;
    fact_present: number;
    no_answer: number;
    first_token_seconds_median?: number | null;
    total_seconds_median?: number | null;
    model_calls_total?: number;
    tokens_total?: number;
    cited_files_in_prompt?: number;
    cited_files_total?: number;
}

export interface ExperimentsBlock {
    title?: string;
    note?: string;
    keep_rule?: string;
    metrics?: string[];
    splits?: Record<string, string>;
    source?: string;
    rows?: ExperimentRow[];
    chat_modes?: {
        title?: string;
        label?: string;
        model?: string;
        questions?: number;
        weak_check?: string;
        stopped_early?: string;
        note?: string;
        source?: string;
        recorded?: Recorded;
        modes?: ChatModeRow[];
    } | null;
}

export interface Summary {
    schema?: number;
    heading_note?: string;
    end_to_end_note?: string;
    retrieval?: RetrievalBlock;
    structural?: StructuralBlock;
    system?: SystemBlock;
    timing?: TimingBlock;
    experiments?: ExperimentsBlock;
    limitations?: LimitationsBlock;
}

export interface BenchmarksResponse {
    available?: boolean;
    heading_note?: string;
    note?: string;
    live?: { embedding_provider?: string; semantic_document_cap?: number };
    summary?: Summary;
}

/* ---- formatting ---- */

/** A metric that was never computed is a dash. 0.000 would claim the embedder scored nothing. */
export const NOT_MEASURED = "—";

export function formatMetric(value: unknown): string {
    return typeof value === "number" && Number.isFinite(value) ? value.toFixed(3) : NOT_MEASURED;
}

export function formatShare(value: unknown): string {
    return typeof value === "number" && Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : NOT_MEASURED;
}

export function formatCount(value: unknown): string {
    return typeof value === "number" && Number.isFinite(value) ? value.toLocaleString("en-US") : NOT_MEASURED;
}

/** "recorded 2026-10-11 · 016c655" — the date and commit every panel has to carry. */
export function recordedLabel(recorded?: Recorded): string {
    if (!recorded?.date && !recorded?.commit) return "";
    return `recorded ${recorded.date ?? NOT_MEASURED} · ${recorded.commit ?? NOT_MEASURED}`;
}

/* ---- Retrieval ---- */

export interface RetrievalRow {
    scope: string;
    questions: number;
    cells: string[];
}

export interface RetrievalSplit {
    id: string;
    questions: number;
    recordedLabel: string;
    source: string;
    rows: RetrievalRow[];
}

export interface RetrievalProvider {
    id: string;
    label: string;
    isDefault: boolean;
    embedder: string;
    note: string;
    splits: RetrievalSplit[];
}

export interface RetrievalView {
    title: string;
    scored: string;
    questionSet: string;
    metrics: string[];
    providers: RetrievalProvider[];
}

export function retrievalView(summary?: Summary): RetrievalView | null {
    const block = summary?.retrieval;
    if (!block?.providers?.length) return null;
    const metrics = block.metrics ?? [];

    return {
        title: block.title ?? "Retrieval",
        scored: block.scored ?? "",
        questionSet: block.question_set ?? "",
        metrics,
        providers: block.providers.map((provider) => ({
            id: provider.id,
            label: provider.label,
            isDefault: provider.default === true,
            embedder: provider.embedder ?? "",
            note: provider.note ?? "",
            splits: (provider.splits ?? []).map((split) => ({
                id: split.id,
                questions: split.questions,
                recordedLabel: recordedLabel(split.recorded),
                source: split.source ?? "",
                rows: (split.rows ?? []).map((row) => ({
                    scope: row.scope,
                    questions: row.questions,
                    cells: metrics.map((metric) => formatMetric(row[metric])),
                })),
            })),
        })),
    };
}

/** The provider shown first: the one the server runs by default. The other is opt-in. */
export function defaultProviderId(view: RetrievalView | null): string {
    if (!view?.providers.length) return "";
    return (view.providers.find((p) => p.isDefault) ?? view.providers[0]).id;
}

/* ---- Structural ---- */

export interface StructuralGroup {
    label: string;
    questions: number;
    precision: string;
    recall: string;
    exact: string;
}

export interface StructuralDisagreement {
    title: string;
    note: string;
    missed: string[];
    unexpected: string[];
}

export interface StructuralView {
    title: string;
    note: string;
    language: string;
    questions: number;
    recordedLabel: string;
    overall: StructuralGroup;
    groups: { heading: string; rows: StructuralGroup[] }[];
    disagreements: StructuralDisagreement[];
    disagreementsNote: string;
}

function group(label: string, value?: Group): StructuralGroup {
    return {
        label,
        questions: value?.questions ?? 0,
        precision: formatMetric(value?.precision),
        recall: formatMetric(value?.recall),
        exact: value?.questions ? `${value.exact}/${value.questions}` : NOT_MEASURED,
    };
}

function groupRows(heading: string, block?: Record<string, Group>): { heading: string; rows: StructuralGroup[] } {
    return { heading, rows: Object.entries(block ?? {}).map(([label, value]) => group(label, value)) };
}

export function structuralView(summary?: Summary): StructuralView | null {
    const block = summary?.structural;
    if (!block?.overall) return null;

    return {
        title: block.title ?? "Structural analysis",
        note: block.note ?? "",
        language: block.language ?? "python",
        questions: block.overall.questions ?? 0,
        recordedLabel: recordedLabel(block.recorded),
        overall: group("all", block.overall),
        groups: [
            groupRows("by split", block.by_split),
            groupRows("by repository", block.by_repo),
            groupRows("by question kind", block.by_kind),
        ].filter((g) => g.rows.length > 0),
        disagreements: (block.disagreements ?? []).map((d) => ({
            title: `${d.repo} · ${d.kind} ${d.subject}`,
            note: d.note ?? "",
            missed: d.ast_found_graph_missed ?? [],
            unexpected: d.graph_found_ast_did_not ?? [],
        })),
        disagreementsNote: block.disagreements_note ?? "",
    };
}

/* ---- System facts ---- */

export interface Fact {
    label: string;
    value: string;
    hint?: string;
}

export interface SystemView {
    title: string;
    endpoints: Fact[];
    endpointsLabel: string;
    live: Fact[];
    callGraph: { repo: string; facts: Fact[] }[];
    callGraphLabel: string;
    callGraphNote: string;
}

export function systemView(summary?: Summary, live?: BenchmarksResponse["live"]): SystemView | null {
    const block = summary?.system;
    if (!block) return null;
    const endpoints = block.endpoints ?? {};
    const callGraph = block.call_graph ?? {};

    return {
        title: block.title ?? "System facts",
        endpointsLabel: recordedLabel(endpoints.recorded),
        endpoints: [
            { label: "Routers", value: formatCount(endpoints.routers) },
            { label: "Routes", value: formatCount(endpoints.routes) },
            { label: "Open", value: formatCount(endpoints.open) },
            { label: "Need a key", value: formatCount(endpoints.need_a_key) },
            { label: "Rate limited", value: formatCount(endpoints.rate_limited) },
        ],
        live: [
            {
                label: "Embedding provider",
                value: live?.embedding_provider ?? NOT_MEASURED,
                hint: "what this server would use for the next index",
            },
            {
                label: "Document cap",
                value: formatCount(live?.semantic_document_cap),
                hint: "applies when the semantic provider is on",
            },
        ],
        callGraphLabel: recordedLabel(callGraph.recorded),
        callGraphNote: callGraph.note ?? "",
        callGraph: Object.entries(callGraph.repos ?? {}).map(([repo, value]) => ({
            repo,
            facts: [
                { label: "Modules", value: formatCount(value.modules) },
                { label: "Definitions", value: formatCount(value.definitions) },
                { label: "Call sites", value: formatCount(value.call_sites) },
                { label: "Resolved", value: `${formatCount(value.resolved)} (${formatShare(value.resolved_share)})` },
                { label: "Import cycle groups", value: formatCount(value.import_cycle_groups) },
            ],
        })),
    };
}

/* ---- Model-free chat timing ---- */

export interface TimingView {
    title: string;
    label: string;
    units: string;
    endToEndNote: string;
    current: { note: string; recordedLabel: string } | null;
    stages: {
        note: string;
        recordedLabel: string;
        columns: string[];
        rows: { case: string; values: string[] }[];
    } | null;
}

export function timingView(summary?: Summary): TimingView | null {
    const block = summary?.timing;
    if (!block) return null;
    const stages = block.stages;

    return {
        title: block.title ?? "Model-free chat timing",
        // Required wording: the panel must say what is not in these numbers.
        label: block.label ?? "excludes the language model",
        units: block.units ?? "",
        endToEndNote: block.end_to_end_note ?? summary?.end_to_end_note ?? "",
        current: block.current
            ? { note: block.current.note ?? "", recordedLabel: recordedLabel(block.current.recorded) }
            : null,
        stages: stages
            ? {
                  note: stages.note ?? "",
                  recordedLabel: recordedLabel(stages.recorded),
                  columns: stages.columns ?? [],
                  rows: (stages.rows ?? []).map((row) => ({
                      case: row.case,
                      values: (row.values ?? []).map((v) => (typeof v === "number" ? `${v}` : NOT_MEASURED)),
                  })),
              }
            : null,
    };
}

/* ---- Limitations and weak checks ---- */

export interface LimitationsView {
    title: string;
    collapsed: boolean;
    smoke: {
        title: string;
        whatItIs: string;
        model: string;
        note: string;
        modes: { mode: string; score: string; cited: string; fact: string; recordedLabel: string }[];
    } | null;
    sampleSizes: { what: string; questions: number; detail: string }[];
    sampleSizeNote: string;
    repos: { name: string; url: string; commit: string }[];
    reposNote: string;
    cap: { value: string; appliesTo: string; note: string; recordedLabel: string } | null;
}

export function limitationsView(summary?: Summary): LimitationsView | null {
    const block = summary?.limitations;
    if (!block) return null;
    const smoke = block.smoke_check;
    const cap = block.document_cap;

    return {
        title: block.title ?? "Limitations and weak checks",
        collapsed: block.collapsed !== false,
        smoke: smoke
            ? {
                  title: smoke.title ?? "Answer smoke check",
                  whatItIs: smoke.what_it_is ?? "",
                  model: smoke.model ?? "",
                  note: smoke.note ?? "",
                  modes: (smoke.modes ?? []).map((m) => ({
                      mode: m.mode,
                      score: `${m.passed}/${m.of}`,
                      cited: `${m.expected_file_cited}/${m.of}`,
                      fact: `${m.fact_present}/${m.of}`,
                      recordedLabel: recordedLabel(m.recorded),
                  })),
              }
            : null,
        sampleSizes: block.sample_sizes ?? [],
        sampleSizeNote: block.sample_size_note ?? "",
        repos: block.repos ?? [],
        reposNote: block.repos_note ?? "",
        cap: cap
            ? {
                  value: formatCount(cap.documents),
                  appliesTo: cap.applies_to ?? "",
                  note: cap.note ?? "",
                  recordedLabel: recordedLabel(cap.recorded),
              }
            : null,
    };
}

/* ---- Experiments ---- */

export interface ExperimentView {
    key: string;
    embedder: string;
    records: string;
    arrangement: string;
    change: string;
    verdict: string;
    /** "chosen" and "baseline" are the two a reader should find first. */
    highlighted: boolean;
    dev: string[];
    heldOut: string[] | null;
    holdout2: string[] | null;
    questions: number;
    recordedLabel: string;
}

export interface ChatModeView {
    title: string;
    label: string;
    model: string;
    questions: number;
    weakCheck: string;
    stoppedEarly: string;
    note: string;
    recordedLabel: string;
    rows: {
        mode: string;
        score: string;
        cited: string;
        fact: string;
        firstToken: string;
        total: string;
        calls: string;
        tokens: string;
        citedInPrompt: string;
    }[];
}

export interface ExperimentsView {
    title: string;
    note: string;
    keepRule: string;
    metrics: string[];
    source: string;
    rows: ExperimentView[];
    chatModes: ChatModeView | null;
}

function cells(scores: ExperimentScores | null | undefined, metrics: string[]): string[] | null {
    if (!scores) return null;
    return metrics.map((metric) => formatMetric(scores[metric]));
}

function seconds(value: unknown): string {
    return typeof value === "number" && Number.isFinite(value) ? `${value.toFixed(2)}s` : NOT_MEASURED;
}

export function experimentsView(summary?: Summary): ExperimentsView | null {
    const block = summary?.experiments;
    if (!block?.rows?.length) return null;
    const metrics = block.metrics ?? [];

    return {
        title: block.title ?? "Experiments",
        note: block.note ?? "",
        keepRule: block.keep_rule ?? "",
        metrics,
        source: block.source ?? "",
        rows: block.rows.map((row) => ({
            key: `${row.embedder}-${row.records}-${row.arrangement}`,
            embedder: row.embedder ?? "",
            records: row.records ?? "",
            arrangement: row.arrangement ?? "",
            change: row.change ?? "",
            verdict: row.verdict ?? "",
            highlighted: row.verdict === "chosen" || row.verdict === "baseline",
            dev: cells(row.dev, metrics) ?? metrics.map(() => NOT_MEASURED),
            heldOut: cells(row.held_out, metrics),
            holdout2: cells(row.holdout2, metrics),
            questions: row.dev?.questions ?? 0,
            recordedLabel: recordedLabel(row.recorded),
        })),
        chatModes: chatModesView(block.chat_modes),
    };
}

function chatModesView(block: ExperimentsBlock["chat_modes"]): ChatModeView | null {
    if (!block?.modes?.length) return null;
    return {
        title: block.title ?? "Chat mode comparison",
        // Required wording: this is one run on one machine, not a benchmark.
        label: block.label ?? "one local run, one model, one question set - not a benchmark",
        model: block.model ?? NOT_MEASURED,
        questions: block.questions ?? 0,
        weakCheck: block.weak_check ?? "",
        stoppedEarly: block.stopped_early ?? "",
        note: block.note ?? "",
        recordedLabel: recordedLabel(block.recorded),
        rows: block.modes.map((row) => ({
            mode: row.mode,
            score: `${row.passed}/${row.runs}`,
            cited: `${row.expected_file_cited}/${row.runs}`,
            fact: `${row.fact_present}/${row.runs}`,
            firstToken: seconds(row.first_token_seconds_median),
            total: seconds(row.total_seconds_median),
            calls: formatCount(row.model_calls_total),
            tokens: formatCount(row.tokens_total),
            citedInPrompt: `${row.cited_files_in_prompt ?? 0}/${row.cited_files_total ?? 0}`,
        })),
    };
}

/* ---- The section as a whole ---- */

export interface BenchmarksView {
    available: boolean;
    /** Shown under the section heading, whether or not there is any data. */
    headingNote: string;
    /** Why there is nothing to show, when there is nothing to show. */
    unavailableNote: string;
    retrieval: RetrievalView | null;
    structural: StructuralView | null;
    system: SystemView | null;
    timing: TimingView | null;
    experiments: ExperimentsView | null;
    limitations: LimitationsView | null;
}

const FALLBACK_HEADING_NOTE = "Recorded results on small fixed question sets, not live accuracy.";
const FALLBACK_UNAVAILABLE = "Recorded results are not available in this build.";

/**
 * The whole section from one response. A response that says `available: false`, one with no
 * summary, and one whose summary is missing a block all produce a view the page can draw: the
 * heading note is always there, and a panel with no data is simply absent.
 */
export function benchmarksView(response?: BenchmarksResponse | null): BenchmarksView {
    const summary = response?.available === true ? response.summary : undefined;

    return {
        available: Boolean(summary),
        headingNote: response?.heading_note || FALLBACK_HEADING_NOTE,
        unavailableNote: summary ? "" : response?.note || FALLBACK_UNAVAILABLE,
        retrieval: retrievalView(summary),
        structural: structuralView(summary),
        system: summary ? systemView(summary, response?.live) : null,
        timing: timingView(summary),
        experiments: experimentsView(summary),
        limitations: limitationsView(summary),
    };
}
