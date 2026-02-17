import React from "react";
import { cn } from "@/lib/utils";

interface AppLayoutProps {
    sidebar: React.ReactNode;
    chat: React.ReactNode;
    panels: React.ReactNode;
}

export default function AppLayout({ sidebar, chat, panels }: AppLayoutProps) {
    return (
        <div className="flex h-screen w-full overflow-hidden bg-background text-foreground font-sans">
            {/* Left Sidebar — collapsible via Ctrl+B */}
            {sidebar && (
                <aside className="w-64 flex-shrink-0 border-r border-border bg-sidebar overflow-y-auto transition-all duration-200">
                    {sidebar}
                </aside>
            )}

            {/* Main Chat Panel */}
            <main className="flex-1 flex flex-col min-w-0 bg-background border-r border-border">
                {chat}
            </main>

            {/* Right Insights/Context Panel */}
            <aside className="w-80 flex-shrink-0 bg-background overflow-y-auto hidden lg:block">
                {panels}
            </aside>
        </div>
    );
}
