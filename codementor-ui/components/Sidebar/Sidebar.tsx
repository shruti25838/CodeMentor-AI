"use client";

import React, { useState, useEffect } from "react";
import {
    ChevronDown,
    Plus,
    Loader2,
    ArrowRight,
    Github,
    FolderOpen,
    Check,
    MessageSquare,
    Code2,
    BarChart3,
} from "lucide-react";
import { cn } from "@/lib/utils";
import RepoTree from "./RepoTree";
import { listRepos, indexRepository, RepoInfo } from "@/lib/api";

export default function Sidebar() {
    const [repos, setRepos] = useState<RepoInfo[]>([]);
    const [activeRepo, setActiveRepo] = useState<string | null>(null);
    const [dropdownOpen, setDropdownOpen] = useState(false);
    const [showAddForm, setShowAddForm] = useState(false);
    const [newRepoUrl, setNewRepoUrl] = useState("");
    const [isIndexing, setIsIndexing] = useState(false);
    const [indexError, setIndexError] = useState<string | null>(null);
    const [treeKey, setTreeKey] = useState(0);

    useEffect(() => {
        const currentId = localStorage.getItem("current_repo_id");
        setActiveRepo(currentId);
        listRepos()
            .then((data) => setRepos(data.repos || []))
            .catch(() => {});
    }, []);

    const switchRepo = (repoId: string) => {
        localStorage.setItem("current_repo_id", repoId);
        setActiveRepo(repoId);
        setDropdownOpen(false);
        setTreeKey((k) => k + 1);
        window.dispatchEvent(new CustomEvent("codeatlas:repo-switched", { detail: { repoId } }));
    };

    const handleAddRepo = async () => {
        if (!newRepoUrl.trim() || isIndexing) return;
        setIndexError(null);
        setIsIndexing(true);
        try {
            const result = await indexRepository(newRepoUrl.trim());
            if (result?.repository_id) {
                const repoName = newRepoUrl.trim().replace(/\/$/, "").split("/").pop() || result.repository_id;
                setRepos((prev) =>
                    prev.some((r) => r.repo_id === result.repository_id)
                        ? prev
                        : [...prev, { repo_id: result.repository_id, name: repoName }]
                );
                switchRepo(result.repository_id);
                setNewRepoUrl("");
                setShowAddForm(false);
            }
        } catch (err: any) {
            setIndexError(err.message || "Failed to index repository.");
        } finally {
            setIsIndexing(false);
        }
    };

    // Topic-based follow-up suggestions
    const defaultRepoStarters = [
        "Explain this repository to me",
        "What is the main entry point?",
        "Where is authentication implemented?",
        "Which files have the most dependencies?",
    ];
    const defaultGeneralStarters = [
        "Explain Big-O notation",
        "How do decorators work in Python?",
        "Design a REST API for a todo app",
        "What is dependency injection?",
    ];

    const followUpMap: Record<string, string[]> = {
        // Repo questions
        explain: ["What design patterns does this repo use?", "Show me the main data flow", "What are the key abstractions?", "How is error handling done?"],
        repository: ["What design patterns does this repo use?", "Show me the main data flow", "What are the key abstractions?", "How is error handling done?"],
        structure: ["Explain the folder structure", "How are modules organized?", "Where are the config files?", "What is the architecture style?"],
        entry: ["Trace the startup sequence", "What gets initialized first?", "How does the app bootstrap?", "What are the CLI arguments?"],
        import: ["Find circular dependencies", "Which module is imported most?", "Show the import hierarchy", "Are there unused imports?"],
        dependency: ["What imports what in this repo?", "Find circular dependencies", "Most connected modules", "Dependency injection explained"],
        // General CS topics
        palindrome: ["Reverse a string in Python", "Check if a linked list is a palindrome", "Time complexity of string reversal", "Common string manipulation problems"],
        decorator: ["Create a custom decorator", "Decorator vs context manager", "functools.wraps explained", "Class-based decorators in Python"],
        "big-o": ["Compare O(n) vs O(log n)", "Space complexity basics", "Amortized time complexity", "Best/worst case analysis"],
        api: ["REST vs GraphQL", "API authentication methods", "Rate limiting strategies", "API versioning best practices"],
        sql: ["SQL JOIN types explained", "Database indexing strategies", "ORM vs raw SQL", "Database normalization forms"],
        async: ["asyncio event loop explained", "Promises vs async/await", "Concurrency vs parallelism", "Error handling in async code"],
        sort: ["Compare sorting algorithms", "When to use which sort?", "Stable vs unstable sorting", "External sorting for large data"],
        auth: ["JWT vs session tokens", "OAuth 2.0 flow explained", "Password hashing best practices", "Role-based access control"],
        pattern: ["Factory pattern example", "Singleton pros and cons", "Strategy pattern use cases", "MVC vs MVVM"],
        test: ["Unit vs integration tests", "Mocking in Python", "Test coverage best practices", "TDD workflow explained"],
        error: ["Try/except best practices", "Custom exception classes", "Error handling patterns", "Logging vs error handling"],
        function: ["List all functions in this repo", "Explain the main function", "Find unused functions", "Function complexity analysis"],
        refactor: ["Code smell detection", "Extract method refactoring", "Reduce cyclomatic complexity", "DRY principle examples"],
        class: ["List all classes in this repo", "What's the class hierarchy?", "Find the largest class", "Are SOLID principles followed?"],
        database: ["How is data stored?", "What ORM is being used?", "Show the data models", "How are migrations handled?"],
        config: ["Where are configs loaded?", "What environment variables are used?", "How is logging configured?", "Are there feature flags?"],
        pipeline: ["Explain the data pipeline", "What are the pipeline stages?", "How is data validated?", "Where are transformations done?"],
        model: ["List all data models", "What fields does each model have?", "How are models related?", "Where is validation logic?"],
    };

    // Fallback starters that are always useful
    const fallbackRepoStarters = [
        "Summarize what this codebase does",
        "What are the most important files?",
        "Show me the architecture overview",
        "What frameworks and libraries are used?",
    ];
    const fallbackGeneralStarters = [
        "What are common design patterns?",
        "Explain SOLID principles",
        "How does garbage collection work?",
        "Compare SQL vs NoSQL databases",
    ];

    const [repoStarters, setRepoStarters] = useState(defaultRepoStarters);
    const [generalStarters, setGeneralStarters] = useState(defaultGeneralStarters);

    // Update starters based on user's last question
    useEffect(() => {
        const handler = (e: Event) => {
            const question = ((e as CustomEvent).detail?.question || "").toLowerCase();
            // Find matching topic
            for (const [keyword, suggestions] of Object.entries(followUpMap)) {
                if (question.includes(keyword)) {
                    setGeneralStarters(suggestions);
                    setRepoStarters(suggestions);
                    return;
                }
            }
            // No match — use smart fallbacks instead of mangling the question
            setGeneralStarters(fallbackGeneralStarters);
            setRepoStarters(fallbackRepoStarters);
        };
        window.addEventListener("codeatlas:query", handler);
        return () => window.removeEventListener("codeatlas:query", handler);
    }, []);

    const isGeneralMode = !activeRepo;
    const activeRepoName = repos.find((r) => r.repo_id === activeRepo)?.name || null;

    /* ============ GENERAL MODE — no repo ============ */
    if (isGeneralMode) {
        return (
            <div className="flex flex-col h-full">
                <div className="px-4 pt-6 pb-4 space-y-4">
                    <div className="flex items-center gap-2">
                        <div className="w-8 h-8 rounded-lg bg-white/5 border border-border flex items-center justify-center">
                            <Code2 className="w-4 h-4 text-accent" />
                        </div>
                        <div>
                            <div className="text-xs font-semibold text-foreground">CodeMentor AI</div>
                            <div className="text-[10px] text-muted">General Mode</div>
                        </div>
                    </div>
                    <p className="text-[11px] text-muted/70 leading-relaxed">
                        Ask any coding question — algorithms, design patterns, debugging, best practices, and more.
                    </p>
                </div>

                <div className="px-4 py-3 border-t border-border space-y-3">
                    <h3 className="text-[10px] font-bold uppercase tracking-widest text-muted/60">
                        Quick Starters
                    </h3>
                    {generalStarters.map((q) => (
                        <button
                            key={q}
                            onClick={() => {
                                window.dispatchEvent(
                                    new CustomEvent("codeatlas:prefill-question", { detail: { question: q } })
                                );
                            }}
                            className="w-full text-left px-2 py-1.5 text-[11px] text-muted hover:text-foreground hover:bg-white/5 rounded transition-colors"
                        >
                            <MessageSquare className="w-3 h-3 inline mr-1.5 text-muted/40" />
                            {q}
                        </button>
                    ))}
                </div>

                <div className="mt-auto px-4 py-4 border-t border-border space-y-3">
                    <a
                        href="/eval"
                        className="w-full flex items-center gap-2 px-2 py-1.5 text-[10px] uppercase tracking-widest font-bold text-muted hover:text-accent hover:bg-white/5 rounded transition-colors"
                    >
                        <BarChart3 className="w-3.5 h-3.5" />
                        Eval Dashboard
                    </a>
                    <p className="text-[10px] uppercase tracking-widest text-muted/40 text-center">
                        General Coding Assistant
                    </p>
                </div>
            </div>
        );
    }

    /* ============ REPO MODE ============ */
    return (
        <div className="flex flex-col h-full">
            <div className="px-3 pt-4 pb-2 space-y-2">
                <h2 className="text-[10px] font-bold uppercase tracking-widest text-muted/60 px-1">
                    Repository
                </h2>

                <button
                    onClick={() => setDropdownOpen(!dropdownOpen)}
                    className="w-full flex items-center justify-between px-2 py-1.5 bg-white/5 rounded border border-white/5 hover:bg-white/10 transition-colors text-left"
                >
                    <span className="text-[11px] mono text-accent truncate">
                        {activeRepoName || activeRepo?.slice(0, 8) + "..."}
                    </span>
                    <ChevronDown className={cn("w-3 h-3 text-muted transition-transform flex-shrink-0", dropdownOpen && "rotate-180")} />
                </button>

                {dropdownOpen && (
                    <div className="bg-card border border-border rounded shadow-lg overflow-hidden animate-in fade-in slide-in-from-top-1 duration-150">
                        {repos.map((repo) => (
                            <button
                                key={repo.repo_id}
                                onClick={() => switchRepo(repo.repo_id)}
                                className={cn(
                                    "w-full flex items-center gap-2 px-3 py-2 text-[11px] mono hover:bg-white/5 transition-colors text-left",
                                    repo.repo_id === activeRepo ? "text-accent" : "text-muted"
                                )}
                            >
                                {repo.repo_id === activeRepo ? (
                                    <Check className="w-3 h-3 text-accent flex-shrink-0" />
                                ) : (
                                    <FolderOpen className="w-3 h-3 text-muted/40 flex-shrink-0" />
                                )}
                                <span className="truncate">{repo.name}</span>
                            </button>
                        ))}
                    </div>
                )}

                {!showAddForm ? (
                    <button
                        onClick={() => setShowAddForm(true)}
                        className="w-full flex items-center justify-center gap-1.5 px-2 py-1.5 text-[10px] uppercase tracking-widest font-bold text-muted hover:text-accent hover:bg-white/5 rounded transition-colors border border-dashed border-white/10"
                    >
                        <Plus className="w-3 h-3" />
                        Index New Repo
                    </button>
                ) : (
                    <div className="space-y-2 p-2 bg-white/[0.03] rounded border border-white/5">
                        <div className="relative">
                            <Github className="absolute left-2 top-1/2 -translate-y-1/2 w-3 h-3 text-muted" />
                            <input
                                type="text"
                                value={newRepoUrl}
                                onChange={(e) => setNewRepoUrl(e.target.value)}
                                onKeyDown={(e) => e.key === "Enter" && handleAddRepo()}
                                placeholder="https://github.com/user/repo"
                                className="w-full bg-card border border-border rounded pl-7 pr-2 py-1.5 text-[11px] mono focus:outline-none focus:border-muted placeholder:text-muted/30"
                                disabled={isIndexing}
                            />
                        </div>
                        {indexError && <p className="text-[10px] text-red-400">{indexError}</p>}
                        <div className="flex gap-1.5">
                            <button
                                onClick={handleAddRepo}
                                disabled={!newRepoUrl.trim() || isIndexing}
                                className="flex-1 flex items-center justify-center gap-1 px-2 py-1 bg-accent text-background rounded text-[10px] font-bold hover:opacity-90 disabled:opacity-50"
                            >
                                {isIndexing ? <Loader2 className="w-3 h-3 animate-spin" /> : <>Index <ArrowRight className="w-3 h-3" /></>}
                            </button>
                            <button
                                onClick={() => { setShowAddForm(false); setIndexError(null); }}
                                className="px-2 py-1 text-[10px] text-muted hover:text-foreground transition-colors"
                                disabled={isIndexing}
                            >
                                Cancel
                            </button>
                        </div>
                    </div>
                )}
            </div>

            {/* Quick Starters for Repo Mode */}
            <div className="px-3 py-3 border-t border-border space-y-2">
                <h3 className="text-[10px] font-bold uppercase tracking-widest text-muted/60 px-1">
                    Quick Starters
                </h3>
                {repoStarters.map((q) => (
                    <button
                        key={q}
                        onClick={() => {
                            window.dispatchEvent(
                                new CustomEvent("codeatlas:prefill-question", { detail: { question: q } })
                            );
                        }}
                        className="w-full text-left px-2 py-1.5 text-[11px] text-muted hover:text-foreground hover:bg-white/5 rounded transition-colors"
                    >
                        <MessageSquare className="w-3 h-3 inline mr-1.5 text-muted/40" />
                        {q}
                    </button>
                ))}
            </div>

            <nav className="flex-1 overflow-y-auto">
                <RepoTree key={treeKey} />
            </nav>

            {/* Footer with eval link */}
            <div className="px-3 py-3 border-t border-border flex-shrink-0">
                <a
                    href="/eval"
                    className="w-full flex items-center gap-2 px-2 py-1.5 text-[10px] uppercase tracking-widest font-bold text-muted hover:text-accent hover:bg-white/5 rounded transition-colors"
                >
                    <BarChart3 className="w-3.5 h-3.5" />
                    Eval Dashboard
                </a>
            </div>
        </div>
    );
}
