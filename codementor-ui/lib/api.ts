const BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export async function askQuestion(question: string, repoId?: string) {
    const response = await fetch(`${BASE_URL}/ask`, {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify({ question, repo_id: repoId }),
    });

    if (!response.ok) {
        const error = await response.json().catch(() => ({}));
        throw new Error(error.detail || "Failed to fetch answer from backend");
    }

    return response.json();
}

/** Stream an answer via SSE — calls back on each event type. */
export async function askQuestionStream(
    question: string,
    repoId: string | undefined,
    callbacks: {
        onToken: (token: string) => void;
        onStatus: (status: string) => void;
        onDone: (data: { citations: string[]; reasoning_steps: string[] }) => void;
        onError: (error: string) => void;
    },
) {
    const response = await fetch(`${BASE_URL}/ask/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, repo_id: repoId }),
    });

    if (!response.ok) {
        const err = await response.json().catch(() => ({}));
        throw new Error(err.detail || "Streaming failed");
    }

    const reader = response.body!.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
            if (line.startsWith("data: ")) {
                try {
                    const data = JSON.parse(line.slice(6));
                    switch (data.type) {
                        case "token":
                            callbacks.onToken(data.content);
                            break;
                        case "status":
                            callbacks.onStatus(data.content);
                            break;
                        case "done":
                            callbacks.onDone(data);
                            break;
                        case "error":
                            callbacks.onError(data.content);
                            break;
                    }
                } catch {
                    /* malformed SSE line — skip */
                }
            }
        }
    }
}

export async function indexRepository(repoUrl: string) {
    const response = await fetch(`${BASE_URL}/analyze-repo`, {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify({ repo_url: repoUrl }),
    });

    if (!response.ok) {
        const error = await response.json().catch(() => ({}));
        throw new Error(error.detail || "Failed to start analysis");
    }

    return response.json();
}

export async function fetchFiles(repoId: string) {
    const response = await fetch(`${BASE_URL}/files`, {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify({ repo_id: repoId }),
    });

    if (!response.ok) {
        throw new Error("Failed to fetch file list");
    }

    return response.json();
}

export async function fetchFileContent(
    repoId: string,
    filePath: string,
): Promise<{ path: string; content: string; language: string; line_count: number }> {
    const response = await fetch(`${BASE_URL}/files/content`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ repo_id: repoId, file_path: filePath }),
    });
    if (!response.ok) throw new Error("Failed to fetch file content");
    return response.json();
}

export async function fetchRepoOverview(repoId: string) {
    const response = await fetch(`${BASE_URL}/repo-overview`, {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify({ repo_id: repoId }),
    });

    if (!response.ok) {
        throw new Error("Failed to fetch repository overview");
    }

    return response.json();
}

export interface RepoInfo {
    repo_id: string;
    name: string;
}

export async function listRepos(): Promise<{ repo_ids: string[]; repos: RepoInfo[] }> {
    const response = await fetch(`${BASE_URL}/repos`);
    if (!response.ok) throw new Error("Failed to list repositories");
    return response.json();
}

export async function fetchDependencyGraph(repoId: string) {
    const response = await fetch(`${BASE_URL}/dependencies/graph`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ repo_id: repoId }),
    });
    if (!response.ok) throw new Error("Failed to fetch dependency graph");
    return response.json();
}

export async function fetchEvalStats() {
    const response = await fetch(`${BASE_URL}/eval/stats`);
    if (!response.ok) throw new Error("Failed to fetch eval stats");
    return response.json();
}
