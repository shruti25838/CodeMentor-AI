/** Small, stable public repo used by the landing page's "Try an example" button. */
export const EXAMPLE_REPO = {
    url: "https://github.com/pallets/itsdangerous",
    name: "itsdangerous",
    questions: [
        "How does Signer create and check a signature?",
        "What is the difference between Serializer and URLSafeTimedSerializer?",
        "How does the code detect that a signed value has expired?",
    ],
};

export const SUGGESTIONS_STORAGE_KEY = "suggested_questions";

/** Remember suggested questions for one repo; the workspace shows them only for that repo. */
export function saveSuggestedQuestions(repoId: string, questions: string[]) {
    localStorage.setItem(SUGGESTIONS_STORAGE_KEY, JSON.stringify({ repoId, questions }));
}

export function readSuggestedQuestions(raw: string | null, repoId: string | null): string[] {
    if (!raw || !repoId) return [];
    try {
        const parsed = JSON.parse(raw);
        return parsed.repoId === repoId && Array.isArray(parsed.questions) ? parsed.questions : [];
    } catch {
        return [];
    }
}
