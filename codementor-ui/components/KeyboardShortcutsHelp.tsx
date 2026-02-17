"use client";

import React, { useState, useEffect } from "react";
import { X, Keyboard } from "lucide-react";

const shortcuts = [
    { keys: ["Ctrl", "K"], description: "Focus chat input" },
    { keys: ["Ctrl", "Enter"], description: "Send message" },
    { keys: ["Ctrl", "/"], description: "Toggle shortcuts help" },
    { keys: ["Ctrl", "E"], description: "Open eval dashboard" },
    { keys: ["Esc"], description: "Close modals / panels" },
    { keys: ["Ctrl", "B"], description: "Toggle sidebar" },
    { keys: ["Enter"], description: "Send message (in chat)" },
    { keys: ["Shift", "Enter"], description: "New line (in chat)" },
];

export default function KeyboardShortcutsHelp() {
    const [visible, setVisible] = useState(false);

    useEffect(() => {
        const handler = (e: KeyboardEvent) => {
            if (e.ctrlKey && e.key === "/") {
                e.preventDefault();
                setVisible((v) => !v);
            }
            if (e.key === "Escape" && visible) {
                setVisible(false);
            }
        };
        window.addEventListener("keydown", handler);
        return () => window.removeEventListener("keydown", handler);
    }, [visible]);

    if (!visible) return null;

    return (
        <div
            className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm animate-in fade-in duration-150"
            onClick={(e) => e.target === e.currentTarget && setVisible(false)}
        >
            <div className="bg-card border border-border rounded-2xl shadow-2xl w-full max-w-md mx-4 overflow-hidden animate-in zoom-in-95 duration-200">
                <div className="flex items-center justify-between px-6 py-4 border-b border-border">
                    <div className="flex items-center gap-2">
                        <Keyboard className="w-4 h-4 text-accent" />
                        <h2 className="text-sm font-semibold text-foreground">
                            Keyboard Shortcuts
                        </h2>
                    </div>
                    <button
                        onClick={() => setVisible(false)}
                        className="p-1 text-muted hover:text-foreground rounded hover:bg-white/5 transition-colors"
                    >
                        <X className="w-4 h-4" />
                    </button>
                </div>

                <div className="px-6 py-4 space-y-3">
                    {shortcuts.map((s, i) => (
                        <div key={i} className="flex items-center justify-between">
                            <span className="text-[12px] text-muted">{s.description}</span>
                            <div className="flex gap-1">
                                {s.keys.map((key) => (
                                    <kbd
                                        key={key}
                                        className="px-2 py-0.5 bg-white/5 border border-white/10 rounded text-[10px] mono text-foreground/80 font-bold"
                                    >
                                        {key}
                                    </kbd>
                                ))}
                            </div>
                        </div>
                    ))}
                </div>

                <div className="px-6 py-3 border-t border-border">
                    <p className="text-[10px] text-muted/50 text-center">
                        Press <kbd className="px-1 py-0.5 bg-white/5 rounded text-[9px] mono">Ctrl + /</kbd> to toggle
                    </p>
                </div>
            </div>
        </div>
    );
}
