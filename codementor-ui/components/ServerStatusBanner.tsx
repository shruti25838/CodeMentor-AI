"use client";

import { useSyncExternalStore } from "react";
import { Loader2, AlertTriangle, X } from "lucide-react";
import {
    dismissUnreachable,
    getServerStatus,
    subscribeServerStatus,
    UNREACHABLE_MESSAGE,
    WAKING_MESSAGE,
} from "@/lib/serverRequest";

/** Shows when any backend request is slow (server waking up) or the server could not be reached. */
export default function ServerStatusBanner() {
    const status = useSyncExternalStore(subscribeServerStatus, getServerStatus, () => "ok" as const);
    if (status === "ok") return null;

    const waking = status === "waking";
    return (
        <div
            role="status"
            aria-live="polite"
            className="fixed top-3 left-1/2 -translate-x-1/2 z-[60] max-w-[calc(100%-2rem)] flex items-center gap-2 px-4 py-2 rounded-lg border border-border bg-card shadow-lg text-[12px] text-foreground/90"
        >
            {waking ? (
                <Loader2 className="w-3.5 h-3.5 animate-spin flex-shrink-0" />
            ) : (
                <AlertTriangle className="w-3.5 h-3.5 text-red-400 flex-shrink-0" />
            )}
            <span>{waking ? WAKING_MESSAGE : UNREACHABLE_MESSAGE}</span>
            {!waking && (
                <button
                    onClick={dismissUnreachable}
                    aria-label="Dismiss"
                    className="ml-1 p-0.5 rounded hover:bg-white/10 text-muted"
                >
                    <X className="w-3.5 h-3.5" />
                </button>
            )}
        </div>
    );
}
