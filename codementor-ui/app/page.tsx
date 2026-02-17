"use client";

import { Github, ArrowRight, Loader2, Search, Database, Code2, FolderOpen } from "lucide-react";
import Link from "next/link";
import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";

import { indexRepository, listRepos, RepoInfo } from "@/lib/api";

export default function Home() {
  const [isIndexing, setIsIndexing] = useState(false);
  const [repoUrl, setRepoUrl] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [step, setStep] = useState(0);
  const [existingRepos, setExistingRepos] = useState<RepoInfo[]>([]);
  const router = useRouter();

  useEffect(() => {
    listRepos()
      .then((data) => setExistingRepos(data.repos || []))
      .catch(() => {});
  }, []);

  const openExistingRepo = (repoId: string) => {
    localStorage.setItem("current_repo_id", repoId);
    router.push("/workspace");
  };

  const handleIndex = async () => {
    if (!repoUrl.trim()) return;
    setError(null);
    setIsIndexing(true);
    setStep(0);

    const steps = [
      "Connecting to backend...",
      "Cloning repository...",
      "Parsing AST nodes...",
      "Building import graph...",
      "Generating vector embeddings...",
      "Finalizing index..."
    ];

    try {
      // Start the real indexing process
      const result = await indexRepository(repoUrl);

      // Simulate visual progress steps (backend returns immediately since indexing is backgrounded)
      let current = 0;
      const interval = setInterval(() => {
        if (current < steps.length - 1) {
          current++;
          setStep(current);
        } else {
          clearInterval(interval);
          if (result && result.repository_id) {
            localStorage.setItem("current_repo_id", result.repository_id);
            // Store the repo name extracted from URL
            const repoName = repoUrl.trim().replace(/\/$/, "").split("/").pop() || result.repository_id;
            localStorage.setItem("current_repo_name", repoName);
          }
          router.push("/workspace");
        }
      }, 800);

    } catch (err: any) {
      setError(err.message || "Failed to index repository. Make sure the backend is running.");
      setIsIndexing(false);
    }
  };

  const steps = [
    "Cloning repository...",
    "Parsing AST nodes...",
    "Building import graph...",
    "Generating vector embeddings...",
    "Finalizing index..."
  ];

  return (
    <div className="flex min-h-screen items-center justify-center bg-background px-4">
      <div className="w-full max-w-sm space-y-8">
        <div className="space-y-2 text-center text-accent">
          <div className="bg-white/5 w-12 h-12 rounded-xl flex items-center justify-center mx-auto mb-4 border border-border">
            <Code2 className="w-6 h-6" />
          </div>
          <h1 className="text-2xl font-semibold tracking-tight">CodeMentor AI</h1>
          <p className="text-sm text-muted">Intelligent codebase indexing and assistance.</p>
        </div>

        {isIndexing ? (
          <div className="space-y-6 pt-4">
            <div className="flex items-center gap-3 text-sm font-medium animate-pulse">
              <Loader2 className="w-4 h-4 animate-spin" />
              <span>{steps[step]}</span>
            </div>
            <div className="h-1 w-full bg-white/5 rounded-full overflow-hidden">
              <div
                className="h-full bg-accent transition-all duration-1000 ease-out"
                style={{ width: `${((step + 1) / steps.length) * 100}%` }}
              />
            </div>
            <div className="grid grid-cols-2 gap-2 text-[10px] uppercase tracking-widest text-muted">
              <div className="flex items-center gap-1.5">
                <Search className="w-3 h-3" /> Search Index
              </div>
              <div className="flex items-center gap-1.5">
                <Database className="w-3 h-3" /> Vector Store
              </div>
            </div>
          </div>
        ) : (
          <div className="space-y-4">
            <div className="relative group">
              <div className="absolute inset-y-0 left-3 flex items-center pointer-events-none text-muted group-focus-within:text-accent transition-colors">
                <Github className="w-4 h-4" />
              </div>
              <input
                type="text"
                value={repoUrl}
                onChange={(e) => setRepoUrl(e.target.value)}
                placeholder="https://github.com/user/repo"
                className="w-full bg-card border border-border rounded-md pl-10 pr-4 py-2.5 text-sm focus:outline-none focus:border-muted transition-colors placeholder:text-muted/30"
              />
            </div>

            {error && (
              <p className="text-[11px] text-red-500 font-medium">{error}</p>
            )}

            <button
              onClick={handleIndex}
              disabled={!repoUrl.trim()}
              className="w-full bg-accent text-background rounded-md py-2.5 text-sm font-medium flex items-center justify-center gap-2 hover:opacity-90 transition-opacity mt-4 shadow-lg shadow-white/5 disabled:opacity-50"
            >
              Index Repository
              <ArrowRight className="w-4 h-4" />
            </button>

            {existingRepos.length > 0 && (
              <>
                <div className="relative flex items-center py-2">
                  <div className="flex-grow border-t border-border" />
                  <span className="px-3 text-[10px] text-muted/50 uppercase tracking-widest">previously indexed</span>
                  <div className="flex-grow border-t border-border" />
                </div>

                <div className="space-y-1.5 max-h-40 overflow-y-auto">
                  {existingRepos.map((repo) => (
                    <button
                      key={repo.repo_id}
                      onClick={() => openExistingRepo(repo.repo_id)}
                      className="w-full flex items-center gap-2 px-3 py-2 text-[12px] mono text-muted hover:text-accent bg-white/[0.03] hover:bg-white/[0.06] rounded border border-white/5 transition-colors text-left"
                    >
                      <FolderOpen className="w-3.5 h-3.5 flex-shrink-0" />
                      <span className="truncate">{repo.name}</span>
                      <ArrowRight className="w-3 h-3 ml-auto flex-shrink-0 opacity-0 group-hover:opacity-100" />
                    </button>
                  ))}
                </div>
              </>
            )}

            <div className="relative flex items-center py-2">
              <div className="flex-grow border-t border-border" />
              <span className="px-3 text-[10px] text-muted/50 uppercase tracking-widest">or</span>
              <div className="flex-grow border-t border-border" />
            </div>

            <button
              onClick={() => {
                localStorage.removeItem("current_repo_id");
                localStorage.removeItem("current_repo_name");
                router.push("/workspace");
              }}
              className="w-full border border-border text-foreground/70 rounded-md py-2.5 text-sm font-medium flex items-center justify-center gap-2 hover:bg-white/5 transition-colors"
            >
              Open Coding Workspace
              <ArrowRight className="w-4 h-4" />
            </button>
          </div>
        )}

        <div className="pt-8 border-t border-border text-center">
          <p className="text-[10px] uppercase tracking-widest text-muted/50">
            Professional AI Developer Environment
          </p>
        </div>
      </div>
    </div>
  );
}
