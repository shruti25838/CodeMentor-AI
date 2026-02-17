"use client";

import React, { useState, useEffect, useCallback } from "react";
import { Info, Layers, GitFork, Clock, ChevronRight } from "lucide-react";
import { cn } from "@/lib/utils";
import { fetchRepoOverview, fetchDependencyGraph } from "@/lib/api";
import VisualGraph from "./VisualGraph";

/* ------------------------------------------------------------------ */
/*  Shared event bus so ChatWindow can push retrieved context here     */
/* ------------------------------------------------------------------ */
type ContextEntry = { file: string; score: string };
type ContextListener = (entries: ContextEntry[]) => void;
const listeners: Set<ContextListener> = new Set();

/** Called by ChatWindow after each answer to update the panel */
export function pushRetrievedContext(entries: ContextEntry[]) {
    listeners.forEach((fn) => fn(entries));
}

/* ------------------------------------------------------------------ */
/*  Main component                                                    */
/* ------------------------------------------------------------------ */
export default function ContextPanel() {
    const [activeTab, setActiveTab] = useState<"context" | "graph" | "history">("context");
    const [overview, setOverview] = useState<any>(null);
    const [isLoading, setIsLoading] = useState(true);

    // Retrieved context from last query
    const [context, setContext] = useState<ContextEntry[]>([]);

    // Dependency graph
    const [graph, setGraph] = useState<{
        nodes: string[];
        edges: { source: string; target: string }[];
    } | null>(null);
    const [graphLoading, setGraphLoading] = useState(false);
    const [graphView, setGraphView] = useState<"visual" | "list">("visual");

    // Query history
    const [history, setHistory] = useState<{ question: string; time: string }[]>([]);

    // Subscribe to context pushes
    useEffect(() => {
        const handler: ContextListener = (entries) => setContext(entries);
        listeners.add(handler);
        return () => {
            listeners.delete(handler);
        };
    }, []);

    // Listen for queries (custom event from ChatWindow)
    useEffect(() => {
        const handler = (e: Event) => {
            const detail = (e as CustomEvent).detail;
            if (detail?.question) {
                setHistory((prev) =>
                    [
                        { question: detail.question, time: new Date().toLocaleTimeString() },
                        ...prev,
                    ].slice(0, 50),
                );
            }
        };
        window.addEventListener("codeatlas:query", handler);
        return () => window.removeEventListener("codeatlas:query", handler);
    }, []);

    // Load overview on mount and on repo switch
    const loadOverview = useCallback(async () => {
        setIsLoading(true);
        setOverview(null);
        setGraph(null);
        setContext([]);
        const repoId = localStorage.getItem("current_repo_id");
        if (!repoId) {
            setIsLoading(false);
            return;
        }
        try {
            const data = await fetchRepoOverview(repoId);
            setOverview(data);
        } catch (err) {
            console.error("Failed to load overview", err);
        } finally {
            setIsLoading(false);
        }
    }, []);

    useEffect(() => {
        loadOverview();
    }, [loadOverview]);

    useEffect(() => {
        const handler = () => {
            loadOverview();
        };
        window.addEventListener("codeatlas:repo-switched", handler);
        return () => window.removeEventListener("codeatlas:repo-switched", handler);
    }, [loadOverview]);

    // Load graph when graph tab is selected
    useEffect(() => {
        if (activeTab !== "graph" || graph) return;
        const repoId = localStorage.getItem("current_repo_id");
        if (!repoId) return;
        setGraphLoading(true);
        fetchDependencyGraph(repoId)
            .then((data) => setGraph(data))
            .catch((err) => console.error("Failed to load graph", err))
            .finally(() => setGraphLoading(false));
    }, [activeTab, graph]);

    const tabs = ["context", "graph", "history"] as const;

    return (
        <div className="flex flex-col h-full bg-background border-l border-border">
            {/* Tabs */}
            <div className="flex border-b border-border bg-background/50">
                {tabs.map((tab) => (
                    <button
                        key={tab}
                        onClick={() => setActiveTab(tab)}
                        className={cn(
                            "px-4 py-3 text-[10px] font-bold uppercase tracking-widest transition-colors",
                            tab === activeTab
                                ? "text-accent border-b border-accent"
                                : "text-muted hover:text-foreground",
                        )}
                    >
                        {tab}
                    </button>
                ))}
            </div>

            <div className="flex-1 overflow-y-auto p-4 space-y-8 text-[#E2E2E2]">
                {/* ===================== CONTEXT TAB ===================== */}
                {activeTab === "context" && (
                    <>
                        {/* Repository Statistics */}
                        <section className="space-y-4">
                            <h3 className="text-[10px] font-bold text-muted uppercase tracking-widest flex items-center gap-2">
                                <Info className="w-3.5 h-3.5" />
                                Repository Statistics
                            </h3>
                            <div className="space-y-3">
                                {isLoading ? (
                                    <div className="text-[10px] text-muted animate-pulse font-bold tracking-widest uppercase">
                                        Scanning metadata...
                                    </div>
                                ) : overview ? (
                                    <>
                                        <p className="text-[11px] text-muted leading-relaxed">
                                            Indexed <strong>{overview.file_count}</strong> files
                                            and <strong>{overview.function_count}</strong>{" "}
                                            functions with{" "}
                                            <strong>{overview.dependency_edges}</strong> import
                                            connections.
                                        </p>
                                        <div className="p-3 bg-white/5 rounded border border-white/5 space-y-2">
                                            {Object.entries(
                                                overview.language_counts || {},
                                            ).map(([lang, count]) => (
                                                <div
                                                    key={lang}
                                                    className="flex justify-between text-[10px]"
                                                >
                                                    <span className="text-muted capitalize">
                                                        {lang}
                                                    </span>
                                                    <span className="text-accent mono font-bold">
                                                        {count as any} files
                                                    </span>
                                                </div>
                                            ))}
                                        </div>
                                    </>
                                ) : (
                                    <p className="text-[11px] text-muted">
                                        No metadata available.
                                    </p>
                                )}
                            </div>
                        </section>

                        {/* Retrieved Context */}
                        <section className="space-y-4">
                            <h3 className="text-[10px] font-bold text-muted uppercase tracking-widest flex items-center gap-2">
                                <Layers className="w-3.5 h-3.5" />
                                Retrieved Context
                            </h3>
                            <div className="space-y-2">
                                {context.length === 0 ? (
                                    <div className="px-2 py-1.5 bg-white/5 rounded border border-white/5 text-[11px] mono text-muted">
                                        Ask a question to see retrieved context...
                                    </div>
                                ) : (
                                    context.map((item, i) => (
                                        <div
                                            key={i}
                                            className="flex items-center justify-between px-2 py-1.5 bg-white/5 rounded border border-white/5 hover:bg-white/10 transition-colors cursor-pointer"
                                            onClick={() => {
                                                window.dispatchEvent(
                                                    new CustomEvent(
                                                        "codeatlas:preview-file",
                                                        {
                                                            detail: {
                                                                path: item.file
                                                                    .split(" (")[0]
                                                                    .trim(),
                                                            },
                                                        },
                                                    ),
                                                );
                                            }}
                                        >
                                            <span
                                                className="text-[11px] mono text-muted truncate max-w-[180px]"
                                                title={item.file}
                                            >
                                                {item.file}
                                            </span>
                                            <span className="text-[9px] font-bold text-accent/60 mono ml-2 flex-shrink-0">
                                                {item.score}
                                            </span>
                                        </div>
                                    ))
                                )}
                            </div>
                        </section>
                    </>
                )}

                {/* ===================== GRAPH TAB ===================== */}
                {activeTab === "graph" && (
                    <section className="space-y-4">
                        <div className="flex items-center justify-between">
                            <h3 className="text-[10px] font-bold text-muted uppercase tracking-widest flex items-center gap-2">
                                <GitFork className="w-3.5 h-3.5" />
                                Dependency Graph
                            </h3>
                            {graph && graph.nodes.length > 0 && (
                                <div className="flex gap-1">
                                    <button
                                        onClick={() => setGraphView("visual")}
                                        className={cn(
                                            "px-2 py-0.5 rounded text-[9px] font-bold uppercase tracking-wider transition-colors",
                                            graphView === "visual"
                                                ? "bg-accent/20 text-accent"
                                                : "text-muted hover:text-foreground bg-white/5",
                                        )}
                                    >
                                        Visual
                                    </button>
                                    <button
                                        onClick={() => setGraphView("list")}
                                        className={cn(
                                            "px-2 py-0.5 rounded text-[9px] font-bold uppercase tracking-wider transition-colors",
                                            graphView === "list"
                                                ? "bg-accent/20 text-accent"
                                                : "text-muted hover:text-foreground bg-white/5",
                                        )}
                                    >
                                        List
                                    </button>
                                </div>
                            )}
                        </div>

                        {graphLoading ? (
                            <div className="text-[10px] text-muted animate-pulse font-bold tracking-widest uppercase">
                                Loading graph...
                            </div>
                        ) : graph && graph.nodes.length > 0 ? (
                            graphView === "visual" ? (
                                <VisualGraph nodes={graph.nodes} edges={graph.edges} />
                            ) : (
                                /* List view (fallback / detail) */
                                <GraphListView graph={graph} />
                            )
                        ) : (
                            <p className="text-[11px] text-muted">
                                No graph data available.
                            </p>
                        )}
                    </section>
                )}

                {/* ===================== HISTORY TAB ===================== */}
                {activeTab === "history" && (
                    <section className="space-y-4">
                        <h3 className="text-[10px] font-bold text-muted uppercase tracking-widest flex items-center gap-2">
                            <Clock className="w-3.5 h-3.5" />
                            Query History
                        </h3>
                        {history.length === 0 ? (
                            <p className="text-[11px] text-muted">
                                No queries yet. Ask a question in the chat.
                            </p>
                        ) : (
                            <div className="space-y-2">
                                {history.map((h, i) => (
                                    <div
                                        key={i}
                                        className="px-2 py-2 bg-white/5 rounded border border-white/5 hover:bg-white/10 transition-colors cursor-pointer"
                                        onClick={() => {
                                            window.dispatchEvent(
                                                new CustomEvent(
                                                    "codeatlas:prefill-question",
                                                    { detail: { question: h.question } },
                                                ),
                                            );
                                        }}
                                    >
                                        <div className="text-[11px] text-foreground/80 leading-snug">
                                            {h.question}
                                        </div>
                                        <div className="text-[9px] text-muted/50 mono mt-1">
                                            {h.time}
                                        </div>
                                    </div>
                                ))}
                            </div>
                        )}
                    </section>
                )}
            </div>
        </div>
    );
}

/* ---- List view sub-component ---- */
function GraphListView({
    graph,
}: {
    graph: { nodes: string[]; edges: { source: string; target: string }[] };
}) {
    const [expandedNode, setExpandedNode] = useState<string | null>(null);

    return (
        <div className="space-y-1">
            {graph.nodes.map((node) => {
                const outgoing = graph.edges.filter((e) => e.source === node);
                const incoming = graph.edges.filter((e) => e.target === node);
                const isExpanded = expandedNode === node;
                return (
                    <div
                        key={node}
                        className="border border-white/5 rounded overflow-hidden"
                    >
                        <button
                            onClick={() =>
                                setExpandedNode(isExpanded ? null : node)
                            }
                            className="w-full flex items-center gap-2 px-2 py-1.5 bg-white/5 hover:bg-white/10 transition-colors text-left"
                        >
                            <ChevronRight
                                className={cn(
                                    "w-3 h-3 text-muted transition-transform",
                                    isExpanded && "rotate-90",
                                )}
                            />
                            <span
                                className="text-[11px] mono text-muted truncate flex-1"
                                title={node}
                            >
                                {node}
                            </span>
                            <span className="text-[9px] text-accent/40 mono flex-shrink-0">
                                {outgoing.length}&darr; {incoming.length}&uarr;
                            </span>
                        </button>
                        {isExpanded &&
                            (outgoing.length > 0 || incoming.length > 0) && (
                                <div className="px-3 py-2 space-y-2 bg-white/[0.02] border-t border-white/5">
                                    {outgoing.length > 0 && (
                                        <div>
                                            <div className="text-[9px] text-accent/60 uppercase tracking-widest mb-1">
                                                Imports
                                            </div>
                                            {outgoing.map((e, i) => (
                                                <div
                                                    key={i}
                                                    className="text-[10px] mono text-muted pl-2"
                                                >
                                                    &rarr; {e.target}
                                                </div>
                                            ))}
                                        </div>
                                    )}
                                    {incoming.length > 0 && (
                                        <div>
                                            <div className="text-[9px] text-accent/60 uppercase tracking-widest mb-1">
                                                Imported By
                                            </div>
                                            {incoming.map((e, i) => (
                                                <div
                                                    key={i}
                                                    className="text-[10px] mono text-muted pl-2"
                                                >
                                                    &larr; {e.source}
                                                </div>
                                            ))}
                                        </div>
                                    )}
                                </div>
                            )}
                    </div>
                );
            })}
        </div>
    );
}
