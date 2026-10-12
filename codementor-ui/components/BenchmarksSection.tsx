"use client";

import React, { useState } from "react";
import {
    Activity,
    ChevronDown,
    Cpu,
    FlaskConical,
    GitCompare,
    LucideIcon,
    Server,
    Target,
    Timer,
} from "lucide-react";

import {
    BenchmarksView,
    ExperimentsView,
    Fact,
    LimitationsView,
    NOT_MEASURED,
    RetrievalView,
    StructuralView,
    SystemView,
    TimingView,
    defaultProviderId,
} from "@/lib/benchmarks";

/* ---- shared shells ---- */

function Panel({
    title,
    icon: Icon,
    sub,
    children,
}: {
    title: string;
    icon: LucideIcon;
    sub?: string;
    children: React.ReactNode;
}) {
    return (
        <section className="p-4 sm:p-5 bg-white/[0.02] border border-white/5 rounded-xl space-y-4">
            <header className="space-y-1">
                <h3 className="text-[11px] font-bold uppercase tracking-widest text-muted/70 flex items-center gap-2">
                    <Icon className="w-3.5 h-3.5 shrink-0" />
                    <span className="break-words">{title}</span>
                </h3>
                {sub && <p className="text-[11px] text-muted leading-relaxed">{sub}</p>}
            </header>
            {children}
        </section>
    );
}

/** Where a number came from and when. Every panel carries one. */
function Stamp({ label }: { label: string }) {
    if (!label) return null;
    return <p className="text-[10px] mono text-muted/60">{label}</p>;
}

/** Tables are wide; on a phone they scroll sideways instead of squashing. */
function Scroller({ children }: { children: React.ReactNode }) {
    return <div className="-mx-4 sm:mx-0 px-4 sm:px-0 overflow-x-auto">{children}</div>;
}

/* ---- Retrieval ---- */

function Retrieval({ view }: { view: RetrievalView }) {
    const [providerId, setProviderId] = useState(defaultProviderId(view));
    const provider = view.providers.find((p) => p.id === providerId) ?? view.providers[0];

    return (
        <Panel title={view.title} icon={Target} sub={view.scored}>
            <div className="flex flex-wrap gap-2">
                {view.providers.map((p) => (
                    <button
                        key={p.id}
                        onClick={() => setProviderId(p.id)}
                        aria-pressed={p.id === provider.id}
                        className={
                            p.id === provider.id
                                ? "px-3 py-1.5 rounded-lg text-[11px] font-bold bg-accent/15 text-accent border border-accent/30"
                                : "px-3 py-1.5 rounded-lg text-[11px] font-bold bg-white/5 text-muted border border-white/10 hover:text-foreground"
                        }
                    >
                        {p.label}
                    </button>
                ))}
            </div>

            <p className="text-[11px] mono text-foreground/70 break-words">{provider.embedder}</p>
            {provider.note && <p className="text-[11px] text-muted leading-relaxed">{provider.note}</p>}

            {provider.splits.map((split) => (
                <div key={split.id} className="space-y-2">
                    <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
                        <span className="text-[11px] font-bold uppercase tracking-widest text-foreground/80">
                            {split.id}
                        </span>
                        <span className="text-[10px] mono text-muted">{split.questions} questions</span>
                        <Stamp label={split.recordedLabel} />
                    </div>
                    <Scroller>
                        <table className="w-full min-w-[480px] text-[11px] mono border border-white/5 rounded-lg">
                            <thead>
                                <tr className="bg-white/[0.03]">
                                    <th className="text-left px-3 py-2 font-bold text-muted/60">repo</th>
                                    <th className="text-right px-3 py-2 font-bold text-muted/60">n</th>
                                    {view.metrics.map((m) => (
                                        <th key={m} className="text-right px-3 py-2 font-bold text-muted/60">
                                            {m}
                                        </th>
                                    ))}
                                </tr>
                            </thead>
                            <tbody>
                                {provider.splits
                                    .find((s) => s.id === split.id)!
                                    .rows.map((row) => (
                                        <tr
                                            key={row.scope}
                                            className={
                                                row.scope === "overall"
                                                    ? "border-t border-white/10 text-foreground font-bold"
                                                    : "border-t border-white/5 text-foreground/80"
                                            }
                                        >
                                            <td className="text-left px-3 py-2">{row.scope}</td>
                                            <td className="text-right px-3 py-2 text-muted">{row.questions}</td>
                                            {row.cells.map((cell, i) => (
                                                <td
                                                    key={view.metrics[i]}
                                                    className={
                                                        cell === NOT_MEASURED
                                                            ? "text-right px-3 py-2 text-muted/40"
                                                            : "text-right px-3 py-2"
                                                    }
                                                >
                                                    {cell}
                                                </td>
                                            ))}
                                        </tr>
                                    ))}
                            </tbody>
                        </table>
                    </Scroller>
                </div>
            ))}
            <p className="text-[10px] text-muted/60 leading-relaxed">{view.questionSet}</p>
        </Panel>
    );
}

/* ---- Structural ---- */

function Structural({ view }: { view: StructuralView }) {
    return (
        <Panel title={view.title} icon={GitCompare} sub={view.note}>
            <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
                <span className="text-[10px] mono text-muted">{view.questions} questions</span>
                <span className="text-[10px] mono text-muted">{view.language} only</span>
                <Stamp label={view.recordedLabel} />
            </div>

            <Scroller>
                <table className="w-full min-w-[420px] text-[11px] mono border border-white/5 rounded-lg">
                    <thead>
                        <tr className="bg-white/[0.03]">
                            <th className="text-left px-3 py-2 font-bold text-muted/60">group</th>
                            <th className="text-right px-3 py-2 font-bold text-muted/60">n</th>
                            <th className="text-right px-3 py-2 font-bold text-muted/60">precision</th>
                            <th className="text-right px-3 py-2 font-bold text-muted/60">recall</th>
                            <th className="text-right px-3 py-2 font-bold text-muted/60">exact</th>
                        </tr>
                    </thead>
                    <tbody>
                        {view.groups.map((group) => (
                            <React.Fragment key={group.heading}>
                                <tr className="border-t border-white/10">
                                    <td colSpan={5} className="px-3 py-1.5 text-[10px] uppercase tracking-widest text-muted/50">
                                        {group.heading}
                                    </td>
                                </tr>
                                {group.rows.map((row) => (
                                    <tr key={group.heading + row.label} className="border-t border-white/5 text-foreground/80">
                                        <td className="text-left px-3 py-2">{row.label}</td>
                                        <td className="text-right px-3 py-2 text-muted">{row.questions}</td>
                                        <td className="text-right px-3 py-2">{row.precision}</td>
                                        <td className="text-right px-3 py-2">{row.recall}</td>
                                        <td className="text-right px-3 py-2">{row.exact}</td>
                                    </tr>
                                ))}
                            </React.Fragment>
                        ))}
                        <tr className="border-t border-white/10 text-foreground font-bold">
                            <td className="text-left px-3 py-2">overall</td>
                            <td className="text-right px-3 py-2">{view.overall.questions}</td>
                            <td className="text-right px-3 py-2">{view.overall.precision}</td>
                            <td className="text-right px-3 py-2">{view.overall.recall}</td>
                            <td className="text-right px-3 py-2">{view.overall.exact}</td>
                        </tr>
                    </tbody>
                </table>
            </Scroller>

            {view.disagreements.length > 0 && (
                <div className="space-y-2">
                    <h4 className="text-[10px] font-bold uppercase tracking-widest text-muted/60">
                        Disagreements ({view.disagreements.length})
                    </h4>
                    {view.disagreementsNote && (
                        <p className="text-[11px] text-muted leading-relaxed">{view.disagreementsNote}</p>
                    )}
                    <ul className="space-y-2">
                        {view.disagreements.map((d) => (
                            <li key={d.title} className="p-3 bg-white/[0.03] border border-white/5 rounded-lg space-y-1">
                                <p className="text-[11px] mono text-foreground/90 break-words">{d.title}</p>
                                {d.note && <p className="text-[11px] text-muted">{d.note}</p>}
                                {d.missed.length > 0 && (
                                    <p className="text-[10px] mono text-muted/70 break-words">
                                        ast found, graph missed: {d.missed.join(", ")}
                                    </p>
                                )}
                                {d.unexpected.length > 0 && (
                                    <p className="text-[10px] mono text-muted/70 break-words">
                                        graph found, ast did not: {d.unexpected.join(", ")}
                                    </p>
                                )}
                            </li>
                        ))}
                    </ul>
                </div>
            )}
        </Panel>
    );
}

/* ---- System facts ---- */

function FactGrid({ facts }: { facts: Fact[] }) {
    return (
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3">
            {facts.map((fact) => (
                <div key={fact.label} className="p-3 bg-white/[0.03] border border-white/5 rounded-lg">
                    <div className="text-lg font-bold mono text-foreground break-words">{fact.value}</div>
                    <div className="text-[9px] uppercase tracking-widest text-muted/60 mt-1">{fact.label}</div>
                    {fact.hint && <div className="text-[10px] text-muted/50 mt-1 leading-snug">{fact.hint}</div>}
                </div>
            ))}
        </div>
    );
}

function System({ view }: { view: SystemView }) {
    return (
        <Panel title={view.title} icon={Server}>
            <div className="space-y-2">
                <h4 className="text-[10px] font-bold uppercase tracking-widest text-muted/60">Routers and routes</h4>
                <FactGrid facts={view.endpoints} />
                <Stamp label={view.endpointsLabel} />
            </div>

            <div className="space-y-2">
                <h4 className="text-[10px] font-bold uppercase tracking-widest text-muted/60">This server, right now</h4>
                <FactGrid facts={view.live} />
            </div>

            <div className="space-y-2">
                <h4 className="text-[10px] font-bold uppercase tracking-widest text-muted/60">Call graph coverage</h4>
                {view.callGraph.map((repo) => (
                    <div key={repo.repo} className="space-y-2">
                        <p className="text-[11px] mono text-foreground/80">{repo.repo}</p>
                        <FactGrid facts={repo.facts} />
                    </div>
                ))}
                {view.callGraphNote && (
                    <p className="text-[11px] text-muted leading-relaxed">{view.callGraphNote}</p>
                )}
                <Stamp label={view.callGraphLabel} />
            </div>
        </Panel>
    );
}

/* ---- Model-free chat timing ---- */

function Timing({ view }: { view: TimingView }) {
    return (
        <Panel title={view.title} icon={Timer}>
            <p className="inline-block px-2 py-1 rounded-md bg-amber-400/10 text-amber-300 text-[11px] font-bold">
                {view.label}
            </p>
            {view.current && (
                <div className="space-y-1">
                    <p className="text-[12px] text-foreground/85 leading-relaxed">{view.current.note}</p>
                    <Stamp label={view.current.recordedLabel} />
                </div>
            )}
            {view.stages && (
                <div className="space-y-2">
                    <Scroller>
                        <table className="w-full min-w-[520px] text-[11px] mono border border-white/5 rounded-lg">
                            <thead>
                                <tr className="bg-white/[0.03]">
                                    <th className="text-left px-3 py-2 font-bold text-muted/60">case</th>
                                    {view.stages.columns.map((c) => (
                                        <th key={c} className="text-right px-3 py-2 font-bold text-muted/60">
                                            {c}
                                        </th>
                                    ))}
                                </tr>
                            </thead>
                            <tbody>
                                {view.stages.rows.map((row) => (
                                    <tr key={row.case} className="border-t border-white/5 text-foreground/80">
                                        <td className="text-left px-3 py-2">{row.case}</td>
                                        {row.values.map((value, i) => (
                                            <td key={view.stages!.columns[i]} className="text-right px-3 py-2">
                                                {value}
                                            </td>
                                        ))}
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </Scroller>
                    {view.units && <p className="text-[10px] text-muted/60">{view.units}</p>}
                    <p className="text-[11px] text-muted leading-relaxed">{view.stages.note}</p>
                    <Stamp label={view.stages.recordedLabel} />
                </div>
            )}
            <p className="text-[11px] text-muted leading-relaxed border-t border-white/5 pt-3">{view.endToEndNote}</p>
        </Panel>
    );
}

/* ---- Limitations and weak checks ---- */

function Limitations({ view }: { view: LimitationsView }) {
    const [open, setOpen] = useState(!view.collapsed);

    return (
        <section className="bg-white/[0.02] border border-white/5 rounded-xl">
            <button
                onClick={() => setOpen((v) => !v)}
                aria-expanded={open}
                className="w-full flex items-center justify-between gap-3 p-4 sm:p-5 text-left"
            >
                <span className="text-[11px] font-bold uppercase tracking-widest text-muted/70 flex items-center gap-2">
                    <Activity className="w-3.5 h-3.5 shrink-0" />
                    {view.title}
                </span>
                <ChevronDown
                    className={open ? "w-4 h-4 text-muted rotate-180 transition-transform" : "w-4 h-4 text-muted transition-transform"}
                />
            </button>

            {open && (
                <div className="px-4 sm:px-5 pb-5 space-y-5">
                    {view.smoke && (
                        <div className="space-y-2">
                            <h4 className="text-[10px] font-bold uppercase tracking-widest text-muted/60">
                                {view.smoke.title}
                            </h4>
                            <p className="text-[11px] text-muted leading-relaxed">{view.smoke.whatItIs}</p>
                            <Scroller>
                                <table className="w-full min-w-[420px] text-[11px] mono border border-white/5 rounded-lg">
                                    <thead>
                                        <tr className="bg-white/[0.03]">
                                            <th className="text-left px-3 py-2 font-bold text-muted/60">mode</th>
                                            <th className="text-right px-3 py-2 font-bold text-muted/60">passed</th>
                                            <th className="text-right px-3 py-2 font-bold text-muted/60">file cited</th>
                                            <th className="text-right px-3 py-2 font-bold text-muted/60">fact present</th>
                                            <th className="text-left px-3 py-2 font-bold text-muted/60">recorded</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {view.smoke.modes.map((mode) => (
                                            <tr key={mode.mode} className="border-t border-white/5 text-foreground/80">
                                                <td className="text-left px-3 py-2">{mode.mode}</td>
                                                <td className="text-right px-3 py-2 text-foreground font-bold">{mode.score}</td>
                                                <td className="text-right px-3 py-2">{mode.cited}</td>
                                                <td className="text-right px-3 py-2">{mode.fact}</td>
                                                <td className="text-left px-3 py-2 text-muted/60">{mode.recordedLabel}</td>
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                            </Scroller>
                            {view.smoke.model && (
                                <p className="text-[10px] mono text-muted/60">model: {view.smoke.model}</p>
                            )}
                            <p className="text-[11px] text-muted leading-relaxed">{view.smoke.note}</p>
                        </div>
                    )}

                    {view.sampleSizes.length > 0 && (
                        <div className="space-y-2">
                            <h4 className="text-[10px] font-bold uppercase tracking-widest text-muted/60">Sample sizes</h4>
                            <ul className="space-y-1">
                                {view.sampleSizes.map((sample) => (
                                    <li key={sample.what} className="text-[11px] text-muted leading-relaxed">
                                        <span className="mono text-foreground/80">
                                            {sample.what}: {sample.questions} questions
                                        </span>{" "}
                                        — {sample.detail}
                                    </li>
                                ))}
                            </ul>
                            <p className="text-[11px] text-muted leading-relaxed">{view.sampleSizeNote}</p>
                        </div>
                    )}

                    {view.repos.length > 0 && (
                        <div className="space-y-2">
                            <h4 className="text-[10px] font-bold uppercase tracking-widest text-muted/60">
                                The two repositories
                            </h4>
                            <ul className="space-y-1">
                                {view.repos.map((repo) => (
                                    <li key={repo.name} className="text-[11px] mono text-muted break-words">
                                        {repo.name} — {repo.url} @ {repo.commit}
                                    </li>
                                ))}
                            </ul>
                            <p className="text-[11px] text-muted leading-relaxed">{view.reposNote}</p>
                        </div>
                    )}

                    {view.cap && (
                        <div className="space-y-2">
                            <h4 className="text-[10px] font-bold uppercase tracking-widest text-muted/60">
                                Document cap
                            </h4>
                            <p className="text-[12px] mono text-foreground/85">
                                {view.cap.value} documents — {view.cap.appliesTo}
                            </p>
                            <p className="text-[11px] text-muted leading-relaxed">{view.cap.note}</p>
                            <Stamp label={view.cap.recordedLabel} />
                        </div>
                    )}
                </div>
            )}
        </section>
    );
}

/* ---- Experiments ---- */

function verdictStyle(verdict: string): string {
    if (verdict === "chosen") return "bg-emerald-400/10 text-emerald-300 border-emerald-400/20";
    if (verdict === "baseline") return "bg-white/10 text-foreground/80 border-white/15";
    if (verdict.startsWith("met the rule")) return "bg-amber-400/10 text-amber-300 border-amber-400/20";
    return "bg-white/5 text-muted border-white/10";
}

function Experiments({ view }: { view: ExperimentsView }) {
    const [showAll, setShowAll] = useState(false);
    const rows = showAll ? view.rows : view.rows.filter((r) => r.highlighted || r.verdict.startsWith("met"));

    return (
        <Panel title={view.title} icon={FlaskConical} sub={view.note}>
            <p className="text-[11px] text-muted leading-relaxed">{view.keepRule}</p>

            <Scroller>
                <table className="w-full min-w-[720px] text-[11px] mono border border-white/5 rounded-lg">
                    <thead>
                        <tr className="bg-white/[0.03]">
                            <th className="text-left px-3 py-2 font-bold text-muted/60">embedder</th>
                            <th className="text-left px-3 py-2 font-bold text-muted/60">change</th>
                            <th className="text-right px-3 py-2 font-bold text-muted/60">n</th>
                            {view.metrics.map((m) => (
                                <th key={"dev-" + m} className="text-right px-3 py-2 font-bold text-muted/60">
                                    dev {m}
                                </th>
                            ))}
                            <th className="text-right px-3 py-2 font-bold text-muted/60">held out</th>
                            <th className="text-right px-3 py-2 font-bold text-muted/60">fresh</th>
                            <th className="text-left px-3 py-2 font-bold text-muted/60">verdict</th>
                        </tr>
                    </thead>
                    <tbody>
                        {rows.map((row) => (
                            <tr
                                key={row.key}
                                className={
                                    row.highlighted
                                        ? "border-t border-white/10 text-foreground"
                                        : "border-t border-white/5 text-foreground/70"
                                }
                            >
                                <td className="text-left px-3 py-2">{row.embedder}</td>
                                <td className="text-left px-3 py-2 break-words max-w-[260px]">
                                    {row.records} / {row.arrangement}
                                </td>
                                <td className="text-right px-3 py-2 text-muted">{row.questions}</td>
                                {row.dev.map((cell, i) => (
                                    <td key={view.metrics[i]} className="text-right px-3 py-2">
                                        {cell}
                                    </td>
                                ))}
                                <td className="text-right px-3 py-2">
                                    {row.heldOut ? (
                                        row.heldOut[1]
                                    ) : (
                                        <span className="text-muted/40">{NOT_MEASURED}</span>
                                    )}
                                </td>
                                <td className="text-right px-3 py-2">
                                    {row.holdout2 ? (
                                        row.holdout2[1]
                                    ) : (
                                        <span className="text-muted/40">{NOT_MEASURED}</span>
                                    )}
                                </td>
                                <td className="text-left px-3 py-2">
                                    <span
                                        className={
                                            "inline-block px-2 py-0.5 rounded-full border text-[10px] " +
                                            verdictStyle(row.verdict)
                                        }
                                    >
                                        {row.verdict}
                                    </span>
                                </td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </Scroller>

            <div className="flex flex-wrap items-center gap-3">
                <button
                    onClick={() => setShowAll((v) => !v)}
                    aria-expanded={showAll}
                    className="px-3 py-1.5 rounded-lg text-[11px] font-bold bg-white/5 text-muted border border-white/10 hover:text-foreground"
                >
                    {showAll ? "Show the kept ones" : "Show all " + view.rows.length + ", rejected included"}
                </button>
                <span className="text-[10px] mono text-muted/60">
                    held out = scored before; fresh = holdout2, scored once. Both columns show MRR@10.
                </span>
            </div>

            {view.chatModes && <ChatModes view={view.chatModes} />}
        </Panel>
    );
}

function ChatModes({ view }: { view: NonNullable<ExperimentsView["chatModes"]> }) {
    return (
        <div className="space-y-2 pt-2 border-t border-white/5">
            <h4 className="text-[10px] font-bold uppercase tracking-widest text-muted/60">{view.title}</h4>
            <p className="inline-block px-2 py-1 rounded-md bg-amber-400/10 text-amber-300 text-[11px] font-bold">
                {view.label}
            </p>
            <p className="text-[11px] text-muted leading-relaxed">{view.weakCheck}</p>
            <Scroller>
                <table className="w-full min-w-[640px] text-[11px] mono border border-white/5 rounded-lg">
                    <thead>
                        <tr className="bg-white/[0.03]">
                            <th className="text-left px-3 py-2 font-bold text-muted/60">mode</th>
                            <th className="text-right px-3 py-2 font-bold text-muted/60">passed</th>
                            <th className="text-right px-3 py-2 font-bold text-muted/60">file cited</th>
                            <th className="text-right px-3 py-2 font-bold text-muted/60">fact</th>
                            <th className="text-right px-3 py-2 font-bold text-muted/60">first token</th>
                            <th className="text-right px-3 py-2 font-bold text-muted/60">total</th>
                            <th className="text-right px-3 py-2 font-bold text-muted/60">calls</th>
                            <th className="text-right px-3 py-2 font-bold text-muted/60">tokens</th>
                            <th className="text-right px-3 py-2 font-bold text-muted/60">cited in prompt</th>
                        </tr>
                    </thead>
                    <tbody>
                        {view.rows.map((row) => (
                            <tr key={row.mode} className="border-t border-white/5 text-foreground/80">
                                <td className="text-left px-3 py-2">{row.mode}</td>
                                <td className="text-right px-3 py-2 text-foreground font-bold">{row.score}</td>
                                <td className="text-right px-3 py-2">{row.cited}</td>
                                <td className="text-right px-3 py-2">{row.fact}</td>
                                <td className="text-right px-3 py-2">{row.firstToken}</td>
                                <td className="text-right px-3 py-2">{row.total}</td>
                                <td className="text-right px-3 py-2">{row.calls}</td>
                                <td className="text-right px-3 py-2">{row.tokens}</td>
                                <td className="text-right px-3 py-2">{row.citedInPrompt}</td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </Scroller>
            <p className="text-[10px] mono text-muted/60">
                model: {view.model} - {view.questions} questions
            </p>
            {view.stoppedEarly && (
                <p className="text-[11px] text-amber-300/80 leading-relaxed">stopped early: {view.stoppedEarly}</p>
            )}
            <p className="text-[11px] text-muted leading-relaxed">{view.note}</p>
            <Stamp label={view.recordedLabel} />
        </div>
    );
}

/* ---- The section ---- */

export default function BenchmarksSection({ view }: { view: BenchmarksView }) {
    return (
        <section>
            <h2 className="text-[10px] font-bold uppercase tracking-widest text-muted/60 mb-1 flex items-center gap-2">
                <Cpu className="w-3.5 h-3.5" />
                Benchmarks
            </h2>
            {/* Required wording. It stays whether or not there is anything to show below it. */}
            <p className="text-[11px] text-muted mb-4 leading-relaxed">{view.headingNote}</p>

            {!view.available ? (
                <div className="p-5 bg-white/[0.02] border border-white/5 rounded-xl text-[12px] text-muted leading-relaxed">
                    {view.unavailableNote}
                </div>
            ) : (
                <div className="space-y-4">
                    {view.retrieval && <Retrieval view={view.retrieval} />}
                    {view.structural && <Structural view={view.structural} />}
                    {view.system && <System view={view.system} />}
                    {view.timing && <Timing view={view.timing} />}
                    {view.experiments && <Experiments view={view.experiments} />}
                    {view.limitations && <Limitations view={view.limitations} />}
                </div>
            )}
        </section>
    );
}
