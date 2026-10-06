"use client";

import React, { useState, useRef, useEffect, useCallback, useSyncExternalStore } from "react";
import { Send, Terminal, Loader2, Lightbulb, Layers } from "lucide-react";
import { cn } from "@/lib/utils";
import MessageBubble from "./MessageBubble";

interface AgentStep {
    name: string;
    detail: string;
}

interface Message {
    role: "user" | "planner" | "analyst" | "mentor" | "memory";
    content: string;
    reasoningSteps?: string[];
    citations?: string[];
    agentSteps?: AgentStep[];
}

import { askQuestionStream, type AskMode } from "@/lib/api";
import { newChatSessionId } from "@/lib/chatSession";
import { pushRetrievedContext } from "@/components/Panels/ContextPanel";
import { readSuggestedQuestions, SUGGESTIONS_STORAGE_KEY } from "@/lib/example";
import { getServerStatus, subscribeServerStatus, WAKING_MESSAGE } from "@/lib/serverRequest";

const subscribeToStorage = (onChange: () => void) => {
    window.addEventListener("storage", onChange);
    return () => window.removeEventListener("storage", onChange);
};
// Returns a JSON string so React can compare snapshots by value.
const readSuggestionsSnapshot = () =>
    JSON.stringify(
        readSuggestedQuestions(
            localStorage.getItem(SUGGESTIONS_STORAGE_KEY),
            localStorage.getItem("current_repo_id"),
        ),
    );

export default function ChatWindow() {
    const [messages, setMessages] = useState<Message[]>([
        { role: "planner", content: "I've initialized the workspace. How can I help you today?" },
    ]);
    const [input, setInput] = useState("");
    const [isThinking, setIsThinking] = useState(false);
    // One id per conversation on screen; the server keeps its last few turns (lib/chatSession.ts).
    const [sessionId] = useState(() => newChatSessionId());
    const [streamStatus, setStreamStatus] = useState<string | null>(null);
    // Fast is the default. Deep runs the planned five-agent pipeline and shows each step.
    const [deepMode, setDeepMode] = useState(false);
    const scrollRef = useRef<HTMLDivElement>(null);
    const inputRef = useRef<HTMLTextAreaElement>(null);

    // Suggested questions saved by the landing page's "Try an example" button, for the current repo only.
    const serverStatus = useSyncExternalStore(subscribeServerStatus, getServerStatus, () => "ok" as const);
    const suggestedQuestions: string[] = JSON.parse(
        useSyncExternalStore(subscribeToStorage, readSuggestionsSnapshot, () => "[]"),
    );

    useEffect(() => {
        if (scrollRef.current) {
            scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
        }
    }, [messages, isThinking]);

    // Listen for quick starter prefill events
    useEffect(() => {
        const handler = (e: Event) => {
            const question = (e as CustomEvent).detail?.question;
            if (question) {
                setInput(question);
                inputRef.current?.focus();
            }
        };
        window.addEventListener("codeatlas:prefill-question", handler);
        return () => window.removeEventListener("codeatlas:prefill-question", handler);
    }, []);

    /* ===== Keyboard Shortcuts ===== */
    useEffect(() => {
        const handler = (e: KeyboardEvent) => {
            // Ctrl+K → Focus chat input
            if (e.ctrlKey && e.key === "k") {
                e.preventDefault();
                inputRef.current?.focus();
            }
            // Ctrl+E → Navigate to eval dashboard
            if (e.ctrlKey && e.key === "e") {
                e.preventDefault();
                window.location.href = "/eval";
            }
        };
        window.addEventListener("keydown", handler);
        return () => window.removeEventListener("keydown", handler);
    }, []);

    const handleSend = useCallback(async () => {
        if (!input.trim() || isThinking) return;

        const userMsg = input.trim();
        setMessages((prev) => [...prev, { role: "user", content: userMsg }]);
        setInput("");
        setIsThinking(true);
        setStreamStatus(null);

        try {
            // Fire query event for history panel
            window.dispatchEvent(
                new CustomEvent("codeatlas:query", { detail: { question: userMsg } }),
            );

            const repoId = localStorage.getItem("current_repo_id") || undefined;

            // Add a placeholder message that will be updated as tokens arrive
            const placeholderIdx = messages.length + 1; // +1 for user msg just added
            let streamedContent = "";

            setMessages((prev) => [
                ...prev,
                { role: "mentor", content: "", reasoningSteps: [], citations: [], agentSteps: [] },
            ]);

            await askQuestionStream(userMsg, repoId, sessionId, {
                onToken: (token: string) => {
                    streamedContent += token;
                    setMessages((prev) => {
                        const updated = [...prev];
                        const last = updated[updated.length - 1];
                        if (last && last.role !== "user") {
                            updated[updated.length - 1] = { ...last, content: streamedContent };
                        }
                        return updated;
                    });
                },
                onStatus: (status: string) => {
                    setStreamStatus(status);
                },
                onAgent: (agent) => {
                    setStreamStatus(`${agent.name}: ${agent.detail}`);
                    setMessages((prev) => {
                        const updated = [...prev];
                        const last = updated[updated.length - 1];
                        if (last && last.role !== "user") {
                            updated[updated.length - 1] = {
                                ...last,
                                agentSteps: [...(last.agentSteps || []), agent],
                            };
                        }
                        return updated;
                    });
                },
                onDone: (data) => {
                    setStreamStatus(null);
                    // Final update with citations and reasoning
                    setMessages((prev) => {
                        const updated = [...prev];
                        const last = updated[updated.length - 1];
                        if (last && last.role !== "user") {
                            updated[updated.length - 1] = {
                                ...last,
                                content: streamedContent,
                                citations: data.citations || [],
                                reasoningSteps: data.reasoning_steps || [],
                            };
                        }
                        return updated;
                    });

                    // Push retrieved context to the right panel
                    const citations = data.citations || [];
                    const contextEntries = citations.map((c: string) => {
                        const parts = c.split(" | ");
                        return { file: parts[0] || c, score: parts.length > 1 ? "cited" : "-" };
                    });
                    pushRetrievedContext(contextEntries);
                },
                onError: (error: string) => {
                    setStreamStatus(null);
                    setMessages((prev) => {
                        const updated = [...prev];
                        const last = updated[updated.length - 1];
                        if (last && last.role !== "user") {
                            updated[updated.length - 1] = {
                                ...last,
                                content: `**Error:** ${error}`,
                            };
                        }
                        return updated;
                    });
                },
            }, deepMode ? "deep" : "fast");
        } catch (err: any) {
            const errMsg =
                typeof err === "string"
                    ? err
                    : err?.message ||
                      "I encountered an issue connecting to the backend. Please ensure the FastAPI server is running.";
            setMessages((prev) => {
                const updated = [...prev];
                const last = updated[updated.length - 1];
                if (last && last.role !== "user") {
                    updated[updated.length - 1] = {
                        ...last,
                        content: `**Error:** ${errMsg}`,
                    };
                } else {
                    updated.push({ role: "mentor", content: `**Error:** ${errMsg}` });
                }
                return updated;
            });
        } finally {
            setIsThinking(false);
            setStreamStatus(null);
        }
    }, [input, isThinking, messages.length, sessionId, deepMode]);

    return (
        <div className="flex flex-col h-full bg-background">
            {/* Header */}
            <div className="flex items-center justify-between px-4 min-h-[48px] border-b border-border bg-background/50 backdrop-blur-sm sticky top-0 z-10">
                <div className="text-[10px] font-bold text-muted uppercase tracking-widest flex items-center gap-2">
                    <Terminal className="w-3.5 h-3.5" />
                    Agentic Chat
                    {streamStatus && (
                        <span className="ml-2 text-accent/80 animate-pulse normal-case tracking-normal font-medium">
                            {streamStatus}
                        </span>
                    )}
                </div>
                <div className="flex items-center gap-1.5 text-[10px] text-muted/60">
                    <button
                        type="button"
                        onClick={() => setDeepMode((on) => !on)}
                        aria-pressed={deepMode}
                        title="Deep analysis runs the full five-agent pipeline and shows each step. Slower."
                        className={cn(
                            "flex items-center gap-1 px-2 py-0.5 rounded border transition-colors font-bold uppercase tracking-widest",
                            deepMode
                                ? "border-accent/50 bg-accent/10 text-accent"
                                : "border-white/10 bg-white/5 text-muted/70 hover:text-foreground",
                        )}
                    >
                        <Layers className="w-3 h-3" />
                        Deep analysis
                    </button>
                    <span className="mx-1 text-muted/20">|</span>
                    <kbd className="px-1.5 py-0.5 bg-white/10 border border-white/10 rounded text-[9px] mono text-foreground/50 font-bold">Ctrl+K</kbd>
                    <span className="text-muted/50">focus</span>
                    <span className="mx-1 text-muted/20">|</span>
                    <kbd className="px-1.5 py-0.5 bg-white/10 border border-white/10 rounded text-[9px] mono text-foreground/50 font-bold">Ctrl+/</kbd>
                    <span className="text-muted/50">shortcuts</span>
                </div>
            </div>

            {/* Messages */}
            <div ref={scrollRef} className="flex-1 overflow-y-auto p-4 space-y-8 scroll-smooth">
                {messages.map((msg, i) => (
                    <MessageBubble
                        key={i}
                        role={msg.role}
                        content={msg.content}
                        reasoningSteps={msg.reasoningSteps}
                        citations={msg.citations}
                        agentSteps={msg.agentSteps}
                    />
                ))}

                {messages.length === 1 && suggestedQuestions.length > 0 && (
                    <div className="max-w-3xl mx-auto space-y-2">
                        <div className="text-[10px] font-bold text-muted uppercase tracking-widest flex items-center gap-1.5">
                            <Lightbulb className="w-3 h-3" />
                            Try asking
                        </div>
                        <div className="flex flex-wrap gap-2">
                            {suggestedQuestions.map((q) => (
                                <button
                                    key={q}
                                    onClick={() => {
                                        setInput(q);
                                        inputRef.current?.focus();
                                    }}
                                    className="text-left text-[12px] px-3 py-1.5 rounded-md border border-border bg-white/[0.03] hover:bg-white/[0.06] hover:text-accent text-foreground/80 transition-colors"
                                >
                                    {q}
                                </button>
                            ))}
                        </div>
                    </div>
                )}

                {isThinking && messages[messages.length - 1]?.content === "" && (
                    <div className="max-w-3xl mx-auto flex items-center gap-2 text-muted animate-pulse">
                        <Loader2 className="w-3.5 h-3.5 animate-spin" />
                        <span className="text-[10px] font-bold uppercase tracking-widest">
                            {serverStatus === "waking" ? WAKING_MESSAGE : streamStatus || "Agent is thinking..."}
                        </span>
                    </div>
                )}
            </div>

            {/* Input */}
            <div className="p-4 bg-background border-t border-border">
                <div className="max-w-3xl mx-auto relative">
                    <textarea
                        ref={inputRef}
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        onKeyDown={(e) => {
                            if (e.key === "Enter" && !e.shiftKey) {
                                e.preventDefault();
                                handleSend();
                            }
                            // Ctrl+Enter also sends
                            if (e.ctrlKey && e.key === "Enter") {
                                e.preventDefault();
                                handleSend();
                            }
                        }}
                        placeholder="Ask anything about the codebase... (Enter to send, Shift + Enter for newline)"
                        className="w-full bg-card border border-border rounded-lg px-4 py-3 pr-12 text-sm focus:outline-none focus:border-muted transition-colors resize-none h-24 placeholder:text-muted/50 mono leading-relaxed"
                    />
                    <button
                        onClick={handleSend}
                        disabled={!input.trim() || isThinking}
                        className="absolute bottom-3 right-3 p-1.5 bg-accent text-background rounded hover:opacity-90 transition-opacity disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                        <Send className="w-4 h-4" />
                    </button>
                </div>
            </div>
        </div>
    );
}
