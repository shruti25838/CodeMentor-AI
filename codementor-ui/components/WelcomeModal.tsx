"use client";

import React, { useState, useEffect } from "react";
import {
    X,
    BrainCircuit,
    Search,
    GitFork,
    Code2,
    MessageSquare,
    Lightbulb,
    Sparkles,
} from "lucide-react";

const capabilities = [
    {
        icon: BrainCircuit,
        title: "Planner Agent",
        desc: "Breaks your question into steps and dynamically selects the right agents to handle each part.",
        repoOnly: false,
    },
    {
        icon: Search,
        title: "Code Retrieval (RAG)",
        desc: "Searches your indexed repo using vector embeddings, reranking, and returns grounded answers with citations.",
        repoOnly: true,
    },
    {
        icon: GitFork,
        title: "Dependency Analysis",
        desc: "Builds and visualizes import graphs so you can see which files depend on what.",
        repoOnly: true,
    },
    {
        icon: Code2,
        title: "Code Generation",
        desc: "Generates code, example usage, and refactoring suggestions grounded in your repo's existing patterns.",
        repoOnly: false,
    },
    {
        icon: MessageSquare,
        title: "Coding Mentor",
        desc: "Explains code, answers architecture questions, reviews logic, and teaches best practices like a senior engineer.",
        repoOnly: false,
    },
    {
        icon: Lightbulb,
        title: "Context Memory",
        desc: "Remembers your conversation context so follow-up questions build on previous answers.",
        repoOnly: false,
    },
];

export default function WelcomeModal() {
    const [visible, setVisible] = useState(false);

    useEffect(() => {
        // Show every time the workspace is entered
        setVisible(true);
    }, []);

    const dismiss = () => {
        setVisible(false);
    };

    if (!visible) return null;

    const hasRepo = !!localStorage.getItem("current_repo_id");

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm animate-in fade-in duration-200">
            <div className="bg-card border border-border rounded-2xl shadow-2xl w-full max-w-2xl mx-4 overflow-hidden animate-in zoom-in-95 duration-200">
                {/* Header */}
                <div className="flex items-center justify-between px-8 py-6 border-b border-border">
                    <div className="flex items-center gap-3">
                        <div className="w-10 h-10 rounded-xl bg-accent/10 flex items-center justify-center border border-accent/20">
                            <Sparkles className="w-5 h-5 text-accent" />
                        </div>
                        <div>
                            <h2 className="text-lg font-semibold text-foreground tracking-tight">
                                Let's Get Started!
                            </h2>
                            <p className="text-[11px] text-muted mt-0.5">
                                {hasRepo
                                    ? "Your repository is indexed and ready to explore."
                                    : "General coding workspace — ask anything about programming."}
                            </p>
                        </div>
                    </div>
                    <button
                        onClick={dismiss}
                        className="p-1.5 text-muted hover:text-foreground transition-colors rounded-lg hover:bg-white/5"
                    >
                        <X className="w-5 h-5" />
                    </button>
                </div>

                {/* Capabilities */}
                <div className="px-8 py-6 space-y-4">
                    <p className="text-[10px] font-bold uppercase tracking-widest text-muted/60">
                        Here's what I can do for you
                    </p>
                    <div className="grid grid-cols-2 gap-3">
                        {capabilities
                            .filter((c) => hasRepo || !c.repoOnly)
                            .map((cap) => (
                                <div
                                    key={cap.title}
                                    className="flex gap-3 p-4 rounded-xl bg-white/[0.03] border border-white/5 hover:border-accent/20 hover:bg-white/[0.05] transition-colors"
                                >
                                    <div className="w-8 h-8 rounded-lg bg-accent/10 flex items-center justify-center flex-shrink-0">
                                        <cap.icon className="w-4 h-4 text-accent/80" />
                                    </div>
                                    <div>
                                        <div className="text-[12px] font-semibold text-foreground/90">
                                            {cap.title}
                                        </div>
                                        <div className="text-[11px] text-muted leading-relaxed mt-0.5">
                                            {cap.desc}
                                        </div>
                                    </div>
                                </div>
                            ))}
                    </div>
                </div>

                {/* Footer */}
                <div className="px-8 py-4 border-t border-border flex items-center justify-between">
                    <p className="text-[10px] text-muted/50 uppercase tracking-widest">
                        {hasRepo ? "Repo mode" : "General mode"} — powered by multi-agent AI
                    </p>
                    <button
                        onClick={dismiss}
                        className="px-6 py-2 bg-accent text-background rounded-lg text-[12px] font-bold hover:opacity-90 transition-opacity shadow-lg shadow-accent/10"
                    >
                        Let's Go
                    </button>
                </div>
            </div>
        </div>
    );
}
