"use client";

import React, { useState, useEffect } from "react";
import AppLayout from "@/components/Layout/AppLayout";
import Sidebar from "@/components/Sidebar/Sidebar";
import ChatWindow from "@/components/Chat/ChatWindow";
import ContextPanel from "@/components/Panels/ContextPanel";
import WelcomeModal from "@/components/WelcomeModal";
import FilePreviewModal from "@/components/FilePreviewModal";
import KeyboardShortcutsHelp from "@/components/KeyboardShortcutsHelp";
import type { FilePreviewRequest } from "@/lib/citations";

export default function WorkspacePage() {
    const [preview, setPreview] = useState<FilePreviewRequest | null>(null);
    const [sidebarVisible, setSidebarVisible] = useState(true);

    // Listen for file preview events from RepoTree, ContextPanel and chat citations
    useEffect(() => {
        const handler = (e: Event) => {
            const detail = (e as CustomEvent<FilePreviewRequest>).detail;
            if (detail?.path) setPreview(detail);
        };
        window.addEventListener("codeatlas:preview-file", handler);
        return () => window.removeEventListener("codeatlas:preview-file", handler);
    }, []);

    // Ctrl+B to toggle sidebar
    useEffect(() => {
        const handler = (e: KeyboardEvent) => {
            if (e.ctrlKey && e.key === "b") {
                e.preventDefault();
                setSidebarVisible((v) => !v);
            }
        };
        window.addEventListener("keydown", handler);
        return () => window.removeEventListener("keydown", handler);
    }, []);

    return (
        <>
            <WelcomeModal />
            <KeyboardShortcutsHelp />
            <FilePreviewModal
                filePath={preview?.path ?? null}
                startLine={preview?.startLine}
                endLine={preview?.endLine}
                onClose={() => setPreview(null)}
            />
            <AppLayout
                sidebar={sidebarVisible ? <Sidebar /> : null}
                chat={<ChatWindow />}
                panels={<ContextPanel />}
            />
        </>
    );
}
