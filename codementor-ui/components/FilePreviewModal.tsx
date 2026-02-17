"use client";

import React, { useState, useEffect } from "react";
import { X, FileCode, Loader2, Copy, Check } from "lucide-react";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { vscDarkPlus } from "react-syntax-highlighter/dist/esm/styles/prism";
import { fetchFileContent } from "@/lib/api";

interface FilePreviewModalProps {
    filePath: string | null;
    onClose: () => void;
}

const LANG_MAP: Record<string, string> = {
    python: "python",
    javascript: "javascript",
    typescript: "typescript",
    java: "java",
    go: "go",
    rust: "rust",
    c: "c",
    cpp: "cpp",
    ruby: "ruby",
    csharp: "csharp",
    php: "php",
    json: "json",
    yaml: "yaml",
    markdown: "markdown",
    bash: "bash",
    text: "text",
    toml: "toml",
    ini: "ini",
    kotlin: "kotlin",
    swift: "swift",
};

export default function FilePreviewModal({ filePath, onClose }: FilePreviewModalProps) {
    const [content, setContent] = useState<string>("");
    const [language, setLanguage] = useState<string>("text");
    const [lineCount, setLineCount] = useState(0);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [copied, setCopied] = useState(false);

    useEffect(() => {
        if (!filePath) return;
        const repoId = localStorage.getItem("current_repo_id");
        if (!repoId) {
            setError("No repository selected");
            return;
        }

        setIsLoading(true);
        setError(null);
        setContent("");

        fetchFileContent(repoId, filePath)
            .then((data) => {
                setContent(data.content);
                setLanguage(LANG_MAP[data.language] || data.language || "text");
                setLineCount(data.line_count);
            })
            .catch((err) => setError(err.message))
            .finally(() => setIsLoading(false));
    }, [filePath]);

    // Close on Escape
    useEffect(() => {
        const handler = (e: KeyboardEvent) => {
            if (e.key === "Escape") onClose();
        };
        window.addEventListener("keydown", handler);
        return () => window.removeEventListener("keydown", handler);
    }, [onClose]);

    if (!filePath) return null;

    const handleCopy = () => {
        navigator.clipboard.writeText(content);
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
    };

    return (
        <div
            className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm animate-in fade-in duration-150"
            onClick={(e) => e.target === e.currentTarget && onClose()}
        >
            <div className="bg-card border border-border rounded-2xl shadow-2xl w-full max-w-4xl mx-4 max-h-[85vh] flex flex-col overflow-hidden animate-in zoom-in-95 duration-200">
                {/* Header */}
                <div className="flex items-center justify-between px-5 py-3 border-b border-border bg-background/50 flex-shrink-0">
                    <div className="flex items-center gap-2 min-w-0">
                        <FileCode className="w-4 h-4 text-accent flex-shrink-0" />
                        <span className="text-[12px] mono text-foreground/80 truncate">
                            {filePath}
                        </span>
                        <span className="text-[10px] text-muted bg-white/5 px-2 py-0.5 rounded-full flex-shrink-0">
                            {lineCount} lines
                        </span>
                    </div>
                    <div className="flex items-center gap-2 flex-shrink-0">
                        <button
                            onClick={handleCopy}
                            className="flex items-center gap-1 px-2 py-1 text-[10px] text-muted hover:text-foreground bg-white/5 hover:bg-white/10 rounded transition-colors"
                        >
                            {copied ? (
                                <>
                                    <Check className="w-3 h-3 text-green-400" /> Copied
                                </>
                            ) : (
                                <>
                                    <Copy className="w-3 h-3" /> Copy
                                </>
                            )}
                        </button>
                        <button
                            onClick={onClose}
                            className="p-1.5 text-muted hover:text-foreground transition-colors rounded-lg hover:bg-white/5"
                        >
                            <X className="w-4 h-4" />
                        </button>
                    </div>
                </div>

                {/* Content */}
                <div className="flex-1 overflow-auto">
                    {isLoading ? (
                        <div className="flex items-center justify-center py-20 gap-2 text-muted">
                            <Loader2 className="w-4 h-4 animate-spin" />
                            <span className="text-[11px] font-bold uppercase tracking-widest">
                                Loading file...
                            </span>
                        </div>
                    ) : error ? (
                        <div className="flex items-center justify-center py-20">
                            <p className="text-[12px] text-red-400">{error}</p>
                        </div>
                    ) : (
                        <SyntaxHighlighter
                            language={language}
                            style={vscDarkPlus}
                            showLineNumbers
                            customStyle={{
                                margin: 0,
                                background: "transparent",
                                padding: "1rem",
                                fontSize: "12px",
                                lineHeight: "1.6",
                            }}
                            lineNumberStyle={{
                                minWidth: "3em",
                                paddingRight: "1em",
                                color: "rgba(255,255,255,0.15)",
                                userSelect: "none",
                            }}
                        >
                            {content}
                        </SyntaxHighlighter>
                    )}
                </div>
            </div>
        </div>
    );
}
