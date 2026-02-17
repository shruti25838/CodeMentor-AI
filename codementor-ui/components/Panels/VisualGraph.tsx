"use client";

import React, { useMemo, useState, useEffect } from "react";
import { Maximize2, Minimize2, ChevronDown, ChevronRight } from "lucide-react";
import { cn } from "@/lib/utils";

interface Edge {
    source: string;
    target: string;
}

interface VisualGraphProps {
    nodes: string[];
    edges: Edge[];
}

/* ================================================================ */
/*  Architecture layer definitions                                   */
/* ================================================================ */

interface LayerDef {
    id: string;
    label: string;
    keywords: string[];
    color: string;        // tailwind-ish gradient start
    colorEnd: string;     // gradient end
    border: string;
    textColor: string;
    icon: string;
    order: number;
}

const LAYERS: LayerDef[] = [
    {
        id: "entry",
        label: "Entry Points",
        keywords: ["main", "app.py", "index", "manage", "wsgi", "asgi", "__main__"],
        color: "#059669",
        colorEnd: "#047857",
        border: "#10B981",
        textColor: "#6EE7B7",
        icon: "🚀",
        order: 0,
    },
    {
        id: "api",
        label: "API / Controllers",
        keywords: ["controller", "route", "router", "endpoint", "api", "views"],
        color: "#2563EB",
        colorEnd: "#1D4ED8",
        border: "#3B82F6",
        textColor: "#93C5FD",
        icon: "🔌",
        order: 1,
    },
    {
        id: "middleware",
        label: "Middleware / Auth",
        keywords: ["middleware", "auth", "security", "permission", "guard"],
        color: "#7C3AED",
        colorEnd: "#6D28D9",
        border: "#8B5CF6",
        textColor: "#C4B5FD",
        icon: "🔐",
        order: 2,
    },
    {
        id: "services",
        label: "Services / Business Logic",
        keywords: ["service", "services", "agent", "orchestrat", "pipeline", "trainer", "engine"],
        color: "#9333EA",
        colorEnd: "#7E22CE",
        border: "#A855F7",
        textColor: "#D8B4FE",
        icon: "⚙️",
        order: 3,
    },
    {
        id: "models",
        label: "Models / Schemas",
        keywords: ["model", "schema", "entity", "dataclass", "type"],
        color: "#DB2777",
        colorEnd: "#BE185D",
        border: "#EC4899",
        textColor: "#F9A8D4",
        icon: "📊",
        order: 4,
    },
    {
        id: "data",
        label: "Data / Storage",
        keywords: ["database", "db", "store", "repository", "retriev", "faiss", "embed", "index"],
        color: "#E11D48",
        colorEnd: "#BE123C",
        border: "#F43F5E",
        textColor: "#FDA4AF",
        icon: "💾",
        order: 5,
    },
    {
        id: "config",
        label: "Configuration",
        keywords: ["config", "setting", "env", "constant"],
        color: "#D97706",
        colorEnd: "#B45309",
        border: "#F59E0B",
        textColor: "#FCD34D",
        icon: "⚡",
        order: 6,
    },
    {
        id: "utils",
        label: "Utilities / Helpers",
        keywords: ["util", "helper", "common", "lib", "tool", "logger", "metric", "observ"],
        color: "#EA580C",
        colorEnd: "#C2410C",
        border: "#F97316",
        textColor: "#FDBA74",
        icon: "🔧",
        order: 7,
    },
    {
        id: "frontend",
        label: "Frontend",
        keywords: ["frontend", "component", "page", "layout", "ui", "view", "template", "dashboard"],
        color: "#0891B2",
        colorEnd: "#0E7490",
        border: "#06B6D4",
        textColor: "#67E8F9",
        icon: "🎨",
        order: 8,
    },
    {
        id: "tests",
        label: "Tests",
        keywords: ["test", "spec", "fixture", "conftest", "mock"],
        color: "#4B5563",
        colorEnd: "#374151",
        border: "#6B7280",
        textColor: "#D1D5DB",
        icon: "🧪",
        order: 9,
    },
];

function classifyNode(node: string): string {
    const lower = node.toLowerCase();
    const parts = lower.split("/");
    const fileName = parts[parts.length - 1] || "";

    for (const layer of LAYERS) {
        for (const kw of layer.keywords) {
            // Check directory names and file name
            if (parts.some((p) => p.includes(kw)) || fileName.includes(kw)) {
                return layer.id;
            }
        }
    }
    return "utils"; // default bucket
}

interface ConnectionDetail {
    source: string; // short file name
    target: string;
}

interface ArchLayer {
    def: LayerDef;
    files: string[];
    incomingFrom: Set<string>;
    outgoingTo: Set<string>;
    edgeCount: Map<string, number>; // layerId → count
    connectionDetails: Map<string, ConnectionDetail[]>; // layerId → file pairs
}

/* ================================================================ */
/*  Architecture Flow Component                                      */
/* ================================================================ */

function ArchitectureFlow({
    nodes,
    edges,
    compact,
}: {
    nodes: string[];
    edges: Edge[];
    compact: boolean;
}) {
    const [expandedLayer, setExpandedLayer] = useState<string | null>(null);

    // Classify nodes into layers
    const layers = useMemo(() => {
        const groups = new Map<string, string[]>();
        nodes.forEach((node) => {
            const layerId = classifyNode(node);
            if (!groups.has(layerId)) groups.set(layerId, []);
            groups.get(layerId)!.push(node);
        });

        // Build inter-layer edges with file-level detail
        const nodeToLayer = new Map<string, string>();
        nodes.forEach((n) => nodeToLayer.set(n, classifyNode(n)));

        const layerEdges = new Map<string, Map<string, number>>();
        // Track specific file connections: "srcLayerId" → "tgtLayerId" → [{source, target}]
        const layerConnDetails = new Map<string, Map<string, ConnectionDetail[]>>();

        const sName = (p: string) => {
            const parts = p.split("/");
            return parts[parts.length - 1] || p;
        };

        edges.forEach((e) => {
            const srcLayer = nodeToLayer.get(e.source);
            const tgtLayer = nodeToLayer.get(e.target);
            if (!srcLayer || !tgtLayer || srcLayer === tgtLayer) return;
            // Counts
            if (!layerEdges.has(srcLayer)) layerEdges.set(srcLayer, new Map());
            const counts = layerEdges.get(srcLayer)!;
            counts.set(tgtLayer, (counts.get(tgtLayer) || 0) + 1);
            // Details
            if (!layerConnDetails.has(srcLayer)) layerConnDetails.set(srcLayer, new Map());
            const details = layerConnDetails.get(srcLayer)!;
            if (!details.has(tgtLayer)) details.set(tgtLayer, []);
            const arr = details.get(tgtLayer)!;
            // Deduplicate
            const srcShort = sName(e.source);
            const tgtShort = sName(e.target);
            if (!arr.some((d) => d.source === srcShort && d.target === tgtShort)) {
                arr.push({ source: srcShort, target: tgtShort });
            }
        });

        const result: ArchLayer[] = [];
        for (const layer of LAYERS) {
            const files = groups.get(layer.id);
            if (!files || files.length === 0) continue;

            const incoming = new Set<string>();
            const outgoing = new Set<string>();
            const edgeCounts = new Map<string, number>();
            const connectionDetails = new Map<string, ConnectionDetail[]>();

            layerEdges.forEach((targets, src) => {
                if (targets.has(layer.id)) incoming.add(src);
            });
            const myEdges = layerEdges.get(layer.id);
            if (myEdges) {
                myEdges.forEach((count, tgt) => {
                    outgoing.add(tgt);
                    edgeCounts.set(tgt, count);
                });
            }
            const myDetails = layerConnDetails.get(layer.id);
            if (myDetails) {
                myDetails.forEach((details, tgt) => {
                    connectionDetails.set(tgt, details);
                });
            }

            result.push({
                def: layer,
                files: files.sort(),
                incomingFrom: incoming,
                outgoingTo: outgoing,
                edgeCount: edgeCounts,
                connectionDetails,
            });
        }

        return result;
    }, [nodes, edges]);

    // Get shortened file name
    const shortName = (path: string) => {
        const parts = path.split("/");
        return parts[parts.length - 1] || path;
    };

    // Get language extension color
    const extColor = (path: string) => {
        const ext = path.split(".").pop() || "";
        const colors: Record<string, string> = {
            py: "#3B82F6",
            js: "#EAB308",
            ts: "#2563EB",
            tsx: "#6366F1",
            java: "#EF4444",
            go: "#06B6D4",
            rs: "#F97316",
            rb: "#DC2626",
            json: "#6B7280",
            yaml: "#D97706",
            md: "#9CA3AF",
        };
        return colors[ext] || "#6B7280";
    };

    if (layers.length === 0) {
        return (
            <p className="text-[11px] text-muted text-center py-8">
                No architecture data available.
            </p>
        );
    }

    return (
        <div className={cn("flex flex-col items-center gap-0", compact ? "px-1" : "px-4")}>
            {layers.map((layer, idx) => {
                const isExpanded = expandedLayer === layer.def.id;
                const maxShow = compact ? 4 : 8;

                return (
                    <React.Fragment key={layer.def.id}>
                        {/* Connection arrow from previous layer */}
                        {idx > 0 && (
                            <div className="flex flex-col items-center py-1">
                                <div
                                    className="w-0.5 rounded-full"
                                    style={{
                                        height: "16px",
                                        background: `linear-gradient(to bottom, ${layers[idx - 1].def.border}60, ${layer.def.border}60)`,
                                    }}
                                />
                                {/* Connection details */}
                                {(() => {
                                    const prev = layers[idx - 1];
                                    const prevId = prev.def.id;
                                    const details = prev.connectionDetails.get(layer.def.id) || [];
                                    const reverseDetails = layer.connectionDetails.get(prevId) || [];
                                    const allDetails = [...details, ...reverseDetails];
                                    if (allDetails.length === 0) return null;

                                    const maxShow = compact ? 3 : 5;
                                    const shown = allDetails.slice(0, maxShow);
                                    const remaining = allDetails.length - maxShow;

                                    return (
                                        <div
                                            className="px-2.5 py-1.5 rounded-lg text-center -my-0.5 z-10 space-y-0.5"
                                            style={{
                                                backgroundColor: `${layer.def.border}10`,
                                                border: `1px solid ${layer.def.border}20`,
                                            }}
                                        >
                                            {shown.map((d, i) => (
                                                <div
                                                    key={i}
                                                    className="flex items-center justify-center gap-1 text-[9px] mono"
                                                >
                                                    <span style={{ color: prev.def.textColor }} className="font-bold">
                                                        {d.source}
                                                    </span>
                                                    <span className="text-muted/40">→</span>
                                                    <span style={{ color: layer.def.textColor }} className="font-bold">
                                                        {d.target}
                                                    </span>
                                                </div>
                                            ))}
                                            {remaining > 0 && (
                                                <div className="text-[8px] text-muted/30 mono">
                                                    +{remaining} more
                                                </div>
                                            )}
                                        </div>
                                    );
                                })()}
                                <div
                                    className="w-0.5 rounded-full"
                                    style={{
                                        height: "12px",
                                        background: `${layer.def.border}50`,
                                    }}
                                />
                                <svg
                                    width="12"
                                    height="8"
                                    viewBox="0 0 12 8"
                                    className="-mt-0.5"
                                >
                                    <path
                                        d="M 1 0 L 6 7 L 11 0"
                                        fill="none"
                                        stroke={layer.def.border}
                                        strokeWidth="1.5"
                                        opacity="0.5"
                                    />
                                </svg>
                            </div>
                        )}

                        {/* Layer card */}
                        <div
                            className={cn(
                                "w-full rounded-xl border overflow-hidden transition-all duration-200 hover:scale-[1.01]",
                                compact ? "max-w-[260px]" : "max-w-[500px]",
                            )}
                            style={{
                                borderColor: `${layer.def.border}40`,
                                background: `linear-gradient(135deg, ${layer.def.color}15, ${layer.def.colorEnd}08)`,
                            }}
                        >
                            {/* Header */}
                            <button
                                onClick={() =>
                                    setExpandedLayer(isExpanded ? null : layer.def.id)
                                }
                                className="w-full flex items-center gap-2 px-3 py-2.5 text-left hover:bg-white/[0.02] transition-colors"
                            >
                                <span className={compact ? "text-sm" : "text-lg"}>
                                    {layer.def.icon}
                                </span>
                                <div className="flex-1 min-w-0">
                                    <div
                                        className={cn(
                                            "font-bold tracking-tight",
                                            compact ? "text-[11px]" : "text-[13px]",
                                        )}
                                        style={{ color: layer.def.textColor }}
                                    >
                                        {layer.def.label}
                                    </div>
                                    <div className="text-[9px] text-muted/50 mono">
                                        {layer.files.length} file
                                        {layer.files.length !== 1 ? "s" : ""}
                                    </div>
                                </div>
                                <div
                                    className="px-2 py-0.5 rounded-full text-[10px] mono font-bold"
                                    style={{
                                        backgroundColor: `${layer.def.color}25`,
                                        color: layer.def.textColor,
                                    }}
                                >
                                    {layer.files.length}
                                </div>
                                {isExpanded ? (
                                    <ChevronDown
                                        className="w-3.5 h-3.5 flex-shrink-0"
                                        style={{ color: layer.def.textColor }}
                                    />
                                ) : (
                                    <ChevronRight
                                        className="w-3.5 h-3.5 flex-shrink-0"
                                        style={{ color: `${layer.def.textColor}80` }}
                                    />
                                )}
                            </button>

                            {/* File list (collapsed: show first few, expanded: show all) */}
                            <div
                                className="px-3 pb-2 flex flex-wrap gap-1"
                                style={{
                                    borderTop: isExpanded
                                        ? `1px solid ${layer.def.border}20`
                                        : "none",
                                    paddingTop: isExpanded ? "8px" : "0",
                                }}
                            >
                                {(isExpanded
                                    ? layer.files
                                    : layer.files.slice(0, maxShow)
                                ).map((file) => (
                                    <button
                                        key={file}
                                        className="flex items-center gap-1 px-1.5 py-0.5 rounded bg-white/[0.04] hover:bg-white/[0.08] transition-colors cursor-pointer"
                                        onClick={(e) => {
                                            e.stopPropagation();
                                            window.dispatchEvent(
                                                new CustomEvent(
                                                    "codeatlas:preview-file",
                                                    { detail: { path: file } },
                                                ),
                                            );
                                        }}
                                        title={file}
                                    >
                                        <div
                                            className="w-1.5 h-1.5 rounded-full flex-shrink-0"
                                            style={{
                                                backgroundColor: extColor(file),
                                            }}
                                        />
                                        <span className="text-[9px] mono text-foreground/60 truncate max-w-[120px]">
                                            {shortName(file)}
                                        </span>
                                    </button>
                                ))}
                                {!isExpanded && layer.files.length > maxShow && (
                                    <span className="text-[9px] text-muted/40 mono self-center pl-1">
                                        +{layer.files.length - maxShow} more
                                    </span>
                                )}
                            </div>
                        </div>
                    </React.Fragment>
                );
            })}
        </div>
    );
}

/* ================================================================ */
/*  Main export with fullscreen support                              */
/* ================================================================ */

export default function VisualGraph({ nodes, edges }: VisualGraphProps) {
    const [fullscreen, setFullscreen] = useState(false);

    useEffect(() => {
        if (!fullscreen) return;
        const handler = (e: KeyboardEvent) => {
            if (e.key === "Escape") setFullscreen(false);
        };
        window.addEventListener("keydown", handler);
        return () => window.removeEventListener("keydown", handler);
    }, [fullscreen]);

    if (nodes.length === 0) {
        return (
            <p className="text-[11px] text-muted text-center py-8">
                No architecture data available.
            </p>
        );
    }

    return (
        <>
            {/* Panel view (compact) */}
            <div className="relative w-full">
                <button
                    onClick={() => setFullscreen(true)}
                    className="absolute top-0 right-0 z-10 flex items-center gap-1 px-2 py-1 bg-white/5 border border-white/10 rounded-lg text-[9px] text-muted hover:text-foreground hover:bg-white/10 transition-colors uppercase tracking-wider font-bold"
                >
                    <Maximize2 className="w-3 h-3" />
                    Expand
                </button>
                <div className="pt-8 pb-4 overflow-y-auto" style={{ maxHeight: "600px" }}>
                    <ArchitectureFlow nodes={nodes} edges={edges} compact={true} />
                </div>
            </div>

            {/* Fullscreen modal */}
            {fullscreen && (
                <div className="fixed inset-0 z-50 bg-background/95 backdrop-blur-sm animate-in fade-in duration-150 overflow-y-auto">
                    <div className="sticky top-0 z-20 bg-background/80 backdrop-blur-sm border-b border-white/5 px-6 py-3 flex items-center gap-4">
                        <button
                            onClick={() => setFullscreen(false)}
                            className="flex items-center gap-1.5 px-3 py-1.5 bg-white/5 border border-white/10 rounded-lg text-[11px] text-muted hover:text-foreground hover:bg-white/10 transition-colors"
                        >
                            <Minimize2 className="w-3.5 h-3.5" />
                            Exit Fullscreen
                        </button>
                        <span className="text-[10px] text-muted/40 uppercase tracking-widest font-bold">
                            Architecture Flow
                        </span>
                        <span className="text-[10px] text-muted/30 mono ml-auto">
                            {nodes.length} files · {edges.length} connections
                        </span>
                    </div>
                    <div className="py-8">
                        <ArchitectureFlow
                            nodes={nodes}
                            edges={edges}
                            compact={false}
                        />
                    </div>
                </div>
            )}
        </>
    );
}
