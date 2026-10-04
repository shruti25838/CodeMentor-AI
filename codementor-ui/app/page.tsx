"use client";

import { Github, ArrowRight, Loader2, Code2, FolderOpen, Sparkles } from "lucide-react";
import Link from "next/link";
import { useState, useEffect, useSyncExternalStore } from "react";
import { useRouter } from "next/navigation";

import { indexRepository, listRepos, RepoInfo } from "@/lib/api";
import { EXAMPLE_REPO, saveSuggestedQuestions } from "@/lib/example";
import { getServerStatus, subscribeServerStatus, WAKING_MESSAGE } from "@/lib/serverRequest";

export default function Home() {
  const [isIndexing, setIsIndexing] = useState(false);
  const [repoUrl, setRepoUrl] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [indexingName, setIndexingName] = useState("");
  const [existingRepos, setExistingRepos] = useState<RepoInfo[]>([]);
  const router = useRouter();
  const serverStatus = useSyncExternalStore(subscribeServerStatus, getServerStatus, () => "ok" as const);

  useEffect(() => {
    listRepos()
      .then((data) => setExistingRepos(data.repos || []))
      .catch(() => {});
  }, []);

  const openExistingRepo = (repoId: string) => {
    localStorage.setItem("current_repo_id", repoId);
    router.push("/workspace");
  };

  const handleIndex = async (url: string = repoUrl, suggestedQuestions: string[] = []) => {
    if (!url.trim()) return;
    const repoName = url.trim().replace(/\/$/, "").split("/").pop() || url.trim();
    setError(null);
    setIndexingName(repoName);
    setIsIndexing(true);

    try {
      // The backend answers only after cloning, parsing and indexing have all finished.
      const result = await indexRepository(url);
      if (!result?.repository_id) throw new Error("The server did not return a repository id. Please try again.");
      localStorage.setItem("current_repo_id", result.repository_id);
      localStorage.setItem("current_repo_name", repoName);
      saveSuggestedQuestions(result.repository_id, suggestedQuestions);
      router.push("/workspace");
    } catch (err: any) {
      setError(err.message || "Failed to index repository. Make sure the backend is running.");
      setIsIndexing(false);
    }
  };

  return (
    <div className="flex min-h-screen items-center justify-center bg-background px-4">
      <div className="w-full max-w-sm space-y-8">
        <div className="space-y-2 text-center text-accent">
          <div className="bg-white/5 w-12 h-12 rounded-xl flex items-center justify-center mx-auto mb-4 border border-border">
            <Code2 className="w-6 h-6" />
          </div>
          <h1 className="text-2xl font-semibold tracking-tight">CodeMentor AI</h1>
          <p className="text-sm text-muted">
            Paste a public GitHub repository and ask questions about its code, with answers that cite the
            files they came from.
          </p>
        </div>

        {isIndexing ? (
          <div className="space-y-2 pt-4" role="status" aria-live="polite">
            <div className="flex items-center gap-3 text-sm font-medium">
              <Loader2 className="w-4 h-4 animate-spin flex-shrink-0" />
              <span>
                {serverStatus === "waking" ? WAKING_MESSAGE : `Cloning, parsing and indexing ${indexingName}`}
              </span>
            </div>
            <p className="text-[11px] text-muted pl-7">
              {serverStatus === "waking"
                ? "Indexing starts as soon as the server is up."
                : "This usually takes under a minute. The workspace opens when it is done."}
            </p>
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
              onClick={() => handleIndex()}
              disabled={!repoUrl.trim()}
              className="w-full bg-accent text-background rounded-md py-2.5 text-sm font-medium flex items-center justify-center gap-2 hover:opacity-90 transition-opacity mt-4 shadow-lg shadow-white/5 disabled:opacity-50"
            >
              Index Repository
              <ArrowRight className="w-4 h-4" />
            </button>

            <button
              onClick={() => handleIndex(EXAMPLE_REPO.url, EXAMPLE_REPO.questions)}
              className="w-full border border-border text-foreground/80 rounded-md py-2.5 text-sm font-medium flex items-center justify-center gap-2 hover:bg-white/5 transition-colors"
            >
              <Sparkles className="w-4 h-4" />
              Try an example
            </button>
            <p className="text-[11px] text-muted text-center">
              Indexes the small public{" "}
              <a
                href={EXAMPLE_REPO.url}
                target="_blank"
                rel="noopener noreferrer"
                className="underline hover:text-accent"
              >
                pallets/{EXAMPLE_REPO.name}
              </a>{" "}
              repository and suggests a few questions to ask.
            </p>

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
      </div>
    </div>
  );
}
