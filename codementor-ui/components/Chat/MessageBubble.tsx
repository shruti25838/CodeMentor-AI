"use client";

import React, { useState } from "react";
import ReactMarkdown from "react-markdown";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { vscDarkPlus } from "react-syntax-highlighter/dist/esm/styles/prism";
import {
    ChevronRight,
    ChevronDown,
    BrainCircuit,
    Quote
} from "lucide-react";
import { cn } from "@/lib/utils";

interface MessageBubbleProps {
    role: "user" | "planner" | "analyst" | "mentor" | "memory";
    content: string;
    reasoningSteps?: string[];
    citations?: string[];
}

const AGENT_LABELS = {
    user: "You",
    planner: "Planner",
    analyst: "Repo Analyst",
    mentor: "Coding Mentor",
    memory: "Context Memory",
};

const Collapsible = ({ title, icon: Icon, children, defaultOpen = false, preview }: { title: string, icon: any, children: React.ReactNode, defaultOpen?: boolean, preview?: string }) => {
    const [isOpen, setIsOpen] = useState(defaultOpen);

    return (
        <div className="my-3 border border-border/40 rounded-lg overflow-hidden bg-white/[0.01]">
            <button
                onClick={() => setIsOpen(!isOpen)}
                className="w-full flex items-center justify-between px-3 py-2 hover:bg-white/[0.03] transition-colors"
            >
                <div className="flex items-center gap-2">
                    <div className="w-5 h-5 flex items-center justify-center rounded bg-white/5">
                        <Icon className="w-3 h-3 text-accent/70" />
                    </div>
                    <span className="text-[10px] font-bold uppercase tracking-widest text-muted/80">{title}</span>
                    {!isOpen && preview && (
                        <span className="text-[10px] text-muted/40 truncate max-w-[200px] font-medium">— {preview}</span>
                    )}
                </div>
                {isOpen ? <ChevronDown className="w-3.5 h-3.5 text-muted" /> : <ChevronRight className="w-3.5 h-3.5 text-muted" />}
            </button>
            {isOpen && (
                <div className="px-3 pb-3 pt-1 border-t border-border/20 text-[12px] animate-in fade-in slide-in-from-top-1 duration-200">
                    {children}
                </div>
            )}
        </div>
    );
};

export default function MessageBubble({ role, content, reasoningSteps, citations }: MessageBubbleProps) {
    const isUser = role === "user";

    // Deduplicate citations and clean them up
    const uniqueCitations = Array.from(new Set(citations || [])).filter(Boolean);

    // Get 2-line preview for reasoning (first 60 chars of first step)
    const reasoningPreview = reasoningSteps?.[0]?.slice(0, 60) + "...";

    return (
        <div className={cn("max-w-3xl mx-auto space-y-2", isUser ? "text-right" : "text-left")}>
            <div className={cn(
                "text-[10px] font-bold uppercase tracking-widest px-1",
                isUser ? "text-muted" : "text-accent/70"
            )}>
                {AGENT_LABELS[role]}
            </div>

            <div className={cn(
                "text-sm leading-relaxed px-4 py-3 rounded-xl border",
                isUser
                    ? "bg-white/5 border-white/5 inline-block text-left"
                    : "bg-card border-border text-foreground/90 shadow-sm"
            )}>
                {/* Reasoning Steps (Planner Dropdown) */}
                {!isUser && reasoningSteps && reasoningSteps.length > 0 && (
                    <Collapsible
                        title="Reasoning"
                        icon={BrainCircuit}
                        preview={reasoningPreview}
                    >
                        <div className="space-y-2 font-mono text-muted/80 leading-relaxed">
                            {reasoningSteps.map((step, i) => (
                                <div key={i} className="flex gap-2">
                                    <span className="text-accent/50">[{i + 1}]</span>
                                    <p>{step}</p>
                                </div>
                            ))}
                        </div>
                    </Collapsible>
                )}

                <ReactMarkdown
                    components={{
                        code({ node, inline, className, children, ...props }: any) {
                            const match = /language-(\w+)/.exec(className || "");
                            const codeString = String(children).replace(/\n$/, "");
                            return !inline && match ? (
                                <div className="rounded-md overflow-hidden my-4 border border-white/5">
                                    <div className="bg-white/5 px-3 py-1 flex items-center justify-between border-b border-white/5">
                                        <span className="text-[10px] uppercase tracking-wider text-muted mono">{match[1]}</span>
                                        <button
                                            className="text-[10px] text-muted hover:text-accent transition-colors"
                                            onClick={() => {
                                                navigator.clipboard.writeText(codeString);
                                                const btn = document.activeElement as HTMLButtonElement;
                                                if (btn) { btn.textContent = "Copied!"; setTimeout(() => { btn.textContent = "Copy"; }, 1500); }
                                            }}
                                        >Copy</button>
                                    </div>
                                    <SyntaxHighlighter
                                        style={vscDarkPlus}
                                        language={match[1]}
                                        PreTag="div"
                                        customStyle={{
                                            margin: 0,
                                            background: "rgba(0,0,0,0.2)",
                                            padding: "1.25rem",
                                            fontSize: "13px",
                                            lineHeight: "1.6",
                                        }}
                                        {...props}
                                    >
                                        {String(children).replace(/\n$/, "")}
                                    </SyntaxHighlighter>
                                </div>
                            ) : (
                                <code className="bg-white/10 px-1.5 py-0.5 rounded mono text-accent text-[12px]" {...props}>
                                    {children}
                                </code>
                            );
                        },
                        p: ({ children }) => <p className="mb-4 last:mb-0">{children}</p>,
                        ul: ({ children }) => <ul className="list-disc ml-4 mb-4 space-y-1.5">{children}</ul>,
                        ol: ({ children }) => <ol className="list-decimal ml-4 mb-4 space-y-1.5">{children}</ol>,
                    }}
                >
                    {content}
                </ReactMarkdown>

                {/* Citations Dropdown */}
                {!isUser && uniqueCitations.length > 0 && (
                    <Collapsible title="Citations" icon={Quote}>
                        <div className="grid gap-2">
                            {uniqueCitations.map((cite, i) => (
                                <div key={i} className="group relative flex items-start gap-2 p-2 rounded hover:bg-white/5 transition-colors border border-transparent hover:border-border/40">
                                    <div className="w-1 h-1 rounded-full bg-accent/40 mt-1.5" />
                                    <div className="flex-1 mono text-[11px] text-muted group-hover:text-foreground/80 break-all leading-tight">
                                        {cite}
                                    </div>
                                </div>
                            ))}
                        </div>
                    </Collapsible>
                )}
            </div>
        </div>
    );
}
