"use client";

import React, { useState, useEffect } from "react";
import {
    FileCode,
    Folder,
    ChevronRight,
    ChevronDown,
    FileText,
    Terminal,
    Settings,
    Scale,
    BrainCircuit,
    Database,
    Eye,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { fetchFiles } from "@/lib/api";

interface FileNode {
    name: string;
    type: "file" | "folder";
    path: string; // full relative path
    children?: FileNode[];
}

const FileIcon = ({ name, className }: { name: string; className?: string }) => {
    const ext = name.split(".").pop()?.toLowerCase();

    if (name.includes("config") || ext === "json" || ext === "yaml" || ext === "yml")
        return <Settings className={cn("w-3.5 h-3.5 text-blue-400/80", className)} />;
    if (ext === "py")
        return <FileCode className={cn("w-3.5 h-3.5 text-blue-300/80", className)} />;
    if (ext === "ts" || ext === "tsx")
        return <FileCode className={cn("w-3.5 h-3.5 text-blue-500/80", className)} />;
    if (ext === "js" || ext === "jsx")
        return <FileCode className={cn("w-3.5 h-3.5 text-yellow-400/80", className)} />;
    if (ext === "java")
        return <FileCode className={cn("w-3.5 h-3.5 text-red-400/80", className)} />;
    if (ext === "go")
        return <FileCode className={cn("w-3.5 h-3.5 text-cyan-400/80", className)} />;
    if (ext === "rs")
        return <FileCode className={cn("w-3.5 h-3.5 text-orange-400/80", className)} />;
    if (ext === "rb")
        return <FileCode className={cn("w-3.5 h-3.5 text-red-500/80", className)} />;
    if (ext === "md")
        return <FileText className={cn("w-3.5 h-3.5 text-muted/80", className)} />;
    if (name.toLowerCase().includes("requirement") || ext === "txt")
        return <Terminal className={cn("w-3.5 h-3.5 text-green-400/80", className)} />;
    if (name.toLowerCase().includes("license"))
        return <Scale className={cn("w-3.5 h-3.5 text-yellow-400/80", className)} />;
    if (ext === "db" || ext === "sqlite")
        return <Database className={cn("w-3.5 h-3.5 text-purple-400/80", className)} />;

    return <FileCode className={cn("w-3.5 h-3.5 text-muted/60", className)} />;
};

const Node = ({ node, depth = 0 }: { node: FileNode; depth?: number }) => {
    const [isOpen, setIsOpen] = useState(depth < 1);
    const isFolder = node.type === "folder";

    const handleClick = () => {
        if (isFolder) {
            setIsOpen(!isOpen);
        } else {
            // Dispatch file preview event with the relative path
            window.dispatchEvent(
                new CustomEvent("codeatlas:preview-file", {
                    detail: { path: node.path },
                }),
            );
        }
    };

    return (
        <div>
            <div
                className={cn(
                    "flex items-center gap-1.5 px-3 py-1 text-[12px] cursor-pointer hover:bg-white/[0.03] select-none transition-colors group",
                    depth > 0 && "ml-3",
                )}
                onClick={handleClick}
                style={{ paddingLeft: `${depth * 8 + 12}px` }}
            >
                <div className="w-3 h-3 flex items-center justify-center -ml-4">
                    {isFolder ? (
                        isOpen ? (
                            <ChevronDown className="w-3 h-3 text-muted group-hover:text-foreground/70" />
                        ) : (
                            <ChevronRight className="w-3 h-3 text-muted group-hover:text-foreground/70" />
                        )
                    ) : null}
                </div>

                {isFolder ? (
                    <Folder className={cn("w-3.5 h-3.5 text-accent/70", isOpen && "text-accent")} />
                ) : (
                    <FileIcon name={node.name} />
                )}
                <span
                    className={cn(
                        "truncate font-medium transition-colors flex-1",
                        isFolder
                            ? "text-foreground/80 group-hover:text-foreground"
                            : "text-foreground/60 group-hover:text-foreground/90 font-mono",
                    )}
                >
                    {node.name}
                </span>

                {/* Preview icon on hover for files */}
                {!isFolder && (
                    <Eye className="w-3 h-3 text-muted/0 group-hover:text-muted/40 transition-colors flex-shrink-0" />
                )}
            </div>

            {isFolder && isOpen && node.children && (
                <div className="relative">
                    <div
                        className="absolute left-[13px] top-0 bottom-0 w-[1px] bg-white/[0.05]"
                        style={{ left: `${depth * 8 + 13}px` }}
                    />
                    {node.children.map((child, i) => (
                        <Node key={i} node={child} depth={depth + 1} />
                    ))}
                </div>
            )}
        </div>
    );
};

export default function RepoTree() {
    const [tree, setTree] = useState<FileNode[]>([]);
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        const loadFiles = async () => {
            const repoId = localStorage.getItem("current_repo_id");
            if (!repoId) {
                setIsLoading(false);
                return;
            }

            try {
                const data = await fetchFiles(repoId);
                const builtTree = buildTree(data.files || []);
                setTree(builtTree);
            } catch (err: any) {
                setError(err.message);
            } finally {
                setIsLoading(false);
            }
        };

        loadFiles();
    }, []);

    const buildTree = (files: any[]): FileNode[] => {
        const root: FileNode[] = [];
        const map: Record<string, FileNode> = {};

        files.forEach((file) => {
            const parts = file.path.replace(/\\/g, "/").split("/").filter(Boolean);
            let currentPath = "";

            parts.forEach((part: string, index: number) => {
                const parentPath = currentPath;
                currentPath = currentPath ? `${currentPath}/${part}` : part;
                const isLast = index === parts.length - 1;

                if (!map[currentPath]) {
                    const newNode: FileNode = {
                        name: part,
                        type: isLast ? "file" : "folder",
                        path: currentPath,
                        children: isLast ? undefined : [],
                    };
                    map[currentPath] = newNode;

                    if (index === 0) {
                        root.push(newNode);
                    } else {
                        map[parentPath].children?.push(newNode);
                    }
                }
            });
        });

        const sortNodes = (nodes: FileNode[]) => {
            nodes.sort((a, b) => {
                if (a.type !== b.type) return a.type === "folder" ? -1 : 1;
                return a.name.localeCompare(b.name);
            });
            nodes.forEach((node) => node.children && sortNodes(node.children));
        };

        sortNodes(root);
        return root;
    };

    if (isLoading)
        return (
            <div className="flex flex-col gap-2 px-4 py-3">
                {[1, 2, 3, 4, 5].map((i) => (
                    <div key={i} className="h-4 bg-white/5 rounded animate-pulse" />
                ))}
            </div>
        );

    if (error)
        return (
            <div className="px-4 py-4 text-[11px] text-red-400 font-mono bg-red-400/5 m-2 rounded border border-red-400/20">
                Error loading explorer.
            </div>
        );

    if (tree.length === 0)
        return (
            <div className="flex flex-col items-center justify-center p-8 text-center gap-3">
                <div className="p-3 bg-white/5 rounded-full">
                    <BrainCircuit className="w-5 h-5 text-muted/40" />
                </div>
                <p className="text-[11px] text-muted font-medium">No files indexed yet.</p>
            </div>
        );

    return (
        <div className="flex flex-col h-full">
            <div className="flex items-center justify-between px-4 py-2 border-b border-border/30 bg-white/[0.02]">
                <span className="text-[10px] font-bold text-muted/60 uppercase tracking-widest">
                    Explorer
                </span>
                <div className="flex items-center gap-1.5">
                    <span className="text-[9px] text-muted/30">click to preview</span>
                    <div className="w-1.5 h-1.5 rounded-full bg-green-500/50" />
                </div>
            </div>
            <div className="flex-1 overflow-y-auto pt-2 pb-8 scrollbar-hide">
                {tree.map((node, i) => (
                    <Node key={i} node={node} />
                ))}
            </div>
        </div>
    );
}
