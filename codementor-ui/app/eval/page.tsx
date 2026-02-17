"use client";

import React, { useState, useEffect } from "react";
import {
    BarChart3,
    Clock,
    Zap,
    FileCode,
    BrainCircuit,
    ArrowLeft,
    RefreshCw,
    Loader2,
    Quote,
    Activity,
    GitFork,
} from "lucide-react";
import Link from "next/link";
import { fetchEvalStats } from "@/lib/api";
import { cn } from "@/lib/utils";

/* ---- Metric Card ---- */
function MetricCard({
    label,
    value,
    sub,
    icon: Icon,
    accent,
}: {
    label: string;
    value: string | number;
    sub?: string;
    icon: any;
    accent?: string;
}) {
    return (
        <div className="p-5 bg-white/[0.03] border border-white/5 rounded-xl hover:border-white/10 transition-colors">
            <div className="flex items-center gap-2 mb-3">
                <div
                    className="w-8 h-8 rounded-lg flex items-center justify-center"
                    style={{ backgroundColor: `${accent || "#22D3EE"}15` }}
                >
                    <Icon className="w-4 h-4" style={{ color: accent || "#22D3EE" }} />
                </div>
                <span className="text-[10px] font-bold uppercase tracking-widest text-muted/60">
                    {label}
                </span>
            </div>
            <div className="text-2xl font-bold text-foreground mono tracking-tight">{value}</div>
            {sub && <p className="text-[11px] text-muted mt-1">{sub}</p>}
        </div>
    );
}

/* ---- Bar ---- */
function Bar({ label, value, max, color }: { label: string; value: number; max: number; color: string }) {
    const pct = max > 0 ? (value / max) * 100 : 0;
    return (
        <div className="flex items-center gap-3">
            <span className="text-[11px] mono text-muted w-24 truncate text-right">{label}</span>
            <div className="flex-1 h-5 bg-white/5 rounded-full overflow-hidden">
                <div
                    className="h-full rounded-full transition-all duration-500"
                    style={{ width: `${Math.max(pct, 2)}%`, backgroundColor: color }}
                />
            </div>
            <span className="text-[11px] mono text-foreground/70 w-12 text-right font-bold">{value}</span>
        </div>
    );
}

/* ---- Main Page ---- */
export default function EvalDashboard() {
    const [stats, setStats] = useState<any>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    const loadStats = async () => {
        setLoading(true);
        setError(null);
        try {
            const data = await fetchEvalStats();
            setStats(data);
        } catch (err: any) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        loadStats();
    }, []);

    const agentColors: Record<string, string> = {
        planner: "#8B5CF6",
        retrieval: "#3B82F6",
        mentor: "#22D3EE",
        validator: "#10B981",
        analyst: "#F59E0B",
        memory: "#EC4899",
    };

    return (
        <div className="min-h-screen bg-background text-foreground">
            {/* Header */}
            <header className="border-b border-border bg-background/80 backdrop-blur-sm sticky top-0 z-10">
                <div className="max-w-6xl mx-auto px-6 py-4 flex items-center justify-between">
                    <div className="flex items-center gap-4">
                        <Link
                            href="/workspace"
                            className="flex items-center gap-1 text-muted hover:text-foreground text-[11px] font-bold uppercase tracking-widest transition-colors"
                        >
                            <ArrowLeft className="w-3.5 h-3.5" />
                            Back
                        </Link>
                        <div className="flex items-center gap-2">
                            <BarChart3 className="w-5 h-5 text-accent" />
                            <h1 className="text-lg font-bold tracking-tight">
                                Evaluation Dashboard
                            </h1>
                        </div>
                    </div>
                    <button
                        onClick={loadStats}
                        disabled={loading}
                        className="flex items-center gap-1.5 px-3 py-1.5 bg-white/5 border border-white/10 rounded-lg text-[11px] text-muted hover:text-foreground hover:bg-white/10 transition-colors disabled:opacity-50"
                    >
                        {loading ? (
                            <Loader2 className="w-3.5 h-3.5 animate-spin" />
                        ) : (
                            <RefreshCw className="w-3.5 h-3.5" />
                        )}
                        Refresh
                    </button>
                </div>
            </header>

            <main className="max-w-6xl mx-auto px-6 py-8 space-y-8">
                {error && (
                    <div className="p-4 bg-red-400/10 border border-red-400/20 rounded-xl text-[12px] text-red-400">
                        {error}
                    </div>
                )}

                {loading && !stats ? (
                    <div className="flex items-center justify-center py-20 gap-2 text-muted">
                        <Loader2 className="w-5 h-5 animate-spin" />
                        <span className="text-sm font-bold uppercase tracking-widest">Loading metrics...</span>
                    </div>
                ) : stats ? (
                    <>
                        {/* ---- Session Metrics ---- */}
                        <section>
                            <h2 className="text-[10px] font-bold uppercase tracking-widest text-muted/60 mb-4 flex items-center gap-2">
                                <Activity className="w-3.5 h-3.5" />
                                Session Performance
                            </h2>
                            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
                                <MetricCard
                                    label="Total Queries"
                                    value={stats.total_queries}
                                    sub="since server started"
                                    icon={Zap}
                                    accent="#22D3EE"
                                />
                                <MetricCard
                                    label="Avg Latency"
                                    value={`${stats.avg_latency_ms}ms`}
                                    sub="mean response time"
                                    icon={Clock}
                                    accent="#10B981"
                                />
                                <MetricCard
                                    label="P95 Latency"
                                    value={`${stats.p95_latency_ms}ms`}
                                    sub="95th percentile"
                                    icon={Clock}
                                    accent="#F59E0B"
                                />
                                <MetricCard
                                    label="Avg Citations"
                                    value={stats.avg_citations}
                                    sub="per query"
                                    icon={Quote}
                                    accent="#8B5CF6"
                                />
                            </div>
                        </section>

                        {/* ---- Agent Usage ---- */}
                        {stats.agent_usage && Object.keys(stats.agent_usage).length > 0 && (
                            <section>
                                <h2 className="text-[10px] font-bold uppercase tracking-widest text-muted/60 mb-4 flex items-center gap-2">
                                    <BrainCircuit className="w-3.5 h-3.5" />
                                    Agent Utilization
                                </h2>
                                <div className="p-5 bg-white/[0.02] border border-white/5 rounded-xl space-y-3">
                                    {(() => {
                                        const maxInvocations = Math.max(
                                            ...Object.values(stats.agent_usage).map(
                                                (a: any) => a.invocations,
                                            ),
                                        );
                                        return Object.entries(stats.agent_usage).map(
                                            ([agent, data]: [string, any]) => (
                                                <Bar
                                                    key={agent}
                                                    label={agent}
                                                    value={data.invocations}
                                                    max={maxInvocations}
                                                    color={agentColors[agent] || "#6B7280"}
                                                />
                                            ),
                                        );
                                    })()}
                                </div>
                            </section>
                        )}

                        {/* ---- Repository Stats ---- */}
                        {stats.repositories && stats.repositories.length > 0 && (
                            <section>
                                <h2 className="text-[10px] font-bold uppercase tracking-widest text-muted/60 mb-4 flex items-center gap-2">
                                    <FileCode className="w-3.5 h-3.5" />
                                    Indexed Repositories
                                </h2>
                                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                                    {stats.repositories.map((repo: any) => (
                                        <div
                                            key={repo.repo_id}
                                            className="p-5 bg-white/[0.02] border border-white/5 rounded-xl space-y-3 hover:border-white/10 transition-colors"
                                        >
                                            <div className="flex items-center gap-2">
                                                <GitFork className="w-4 h-4 text-accent" />
                                                <span className="text-[13px] font-semibold text-foreground">
                                                    {repo.name}
                                                </span>
                                            </div>
                                            <div className="grid grid-cols-3 gap-3 text-center">
                                                <div>
                                                    <div className="text-lg font-bold mono text-foreground">
                                                        {repo.file_count}
                                                    </div>
                                                    <div className="text-[9px] text-muted uppercase tracking-widest">
                                                        Files
                                                    </div>
                                                </div>
                                                <div>
                                                    <div className="text-lg font-bold mono text-foreground">
                                                        {repo.function_count}
                                                    </div>
                                                    <div className="text-[9px] text-muted uppercase tracking-widest">
                                                        Functions
                                                    </div>
                                                </div>
                                                <div>
                                                    <div className="text-lg font-bold mono text-foreground">
                                                        {repo.dependency_edges}
                                                    </div>
                                                    <div className="text-[9px] text-muted uppercase tracking-widest">
                                                        Edges
                                                    </div>
                                                </div>
                                            </div>
                                            {/* Language breakdown */}
                                            {repo.languages && Object.keys(repo.languages).length > 0 && (
                                                <div className="flex flex-wrap gap-2 pt-2 border-t border-white/5">
                                                    {Object.entries(repo.languages).map(
                                                        ([lang, count]: [string, any]) => (
                                                            <span
                                                                key={lang}
                                                                className="px-2 py-0.5 bg-white/5 rounded-full text-[10px] mono text-muted"
                                                            >
                                                                {lang}{" "}
                                                                <strong className="text-foreground/70">
                                                                    {count}
                                                                </strong>
                                                            </span>
                                                        ),
                                                    )}
                                                </div>
                                            )}
                                        </div>
                                    ))}
                                </div>
                            </section>
                        )}

                        {/* ---- Recent Queries ---- */}
                        {stats.recent_queries && stats.recent_queries.length > 0 && (
                            <section>
                                <h2 className="text-[10px] font-bold uppercase tracking-widest text-muted/60 mb-4 flex items-center gap-2">
                                    <Clock className="w-3.5 h-3.5" />
                                    Recent Queries
                                </h2>
                                <div className="border border-white/5 rounded-xl overflow-hidden">
                                    <table className="w-full text-[12px]">
                                        <thead>
                                            <tr className="bg-white/[0.03] border-b border-white/5">
                                                <th className="text-left px-4 py-2 text-[10px] uppercase tracking-widest text-muted/60 font-bold">
                                                    Question
                                                </th>
                                                <th className="text-right px-4 py-2 text-[10px] uppercase tracking-widest text-muted/60 font-bold">
                                                    Latency
                                                </th>
                                                <th className="text-right px-4 py-2 text-[10px] uppercase tracking-widest text-muted/60 font-bold">
                                                    Citations
                                                </th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            {stats.recent_queries
                                                .slice()
                                                .reverse()
                                                .map((q: any, i: number) => (
                                                    <tr
                                                        key={i}
                                                        className="border-b border-white/5 last:border-0 hover:bg-white/[0.02] transition-colors"
                                                    >
                                                        <td className="px-4 py-2.5 text-foreground/80 truncate max-w-[400px]">
                                                            {q.question}
                                                        </td>
                                                        <td className="px-4 py-2.5 text-right mono text-muted">
                                                            <span
                                                                className={cn(
                                                                    "px-2 py-0.5 rounded-full text-[10px] font-bold",
                                                                    q.latency_ms < 2000
                                                                        ? "bg-green-400/10 text-green-400"
                                                                        : q.latency_ms < 5000
                                                                          ? "bg-yellow-400/10 text-yellow-400"
                                                                          : "bg-red-400/10 text-red-400",
                                                                )}
                                                            >
                                                                {Math.round(q.latency_ms)}ms
                                                            </span>
                                                        </td>
                                                        <td className="px-4 py-2.5 text-right mono text-muted font-bold">
                                                            {q.citations}
                                                        </td>
                                                    </tr>
                                                ))}
                                        </tbody>
                                    </table>
                                </div>
                            </section>
                        )}
                    </>
                ) : null}
            </main>
        </div>
    );
}
