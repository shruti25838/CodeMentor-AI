/**
 * What the chat window shows while it is waiting, from the stream's status events.
 *
 * The wait a visitor sits through is mostly the model. The server now sends a status event
 * the moment retrieval has finished and before the model is asked anything, carrying the
 * stage name and the files that are about to be cited, so the page can say *Found 3 files*
 * and name them instead of showing a spinner that says nothing for several seconds.
 *
 * Two rules this module exists to keep honest:
 *
 *   - **Only show files the server sent.** `files` arrives only on a path where the server
 *     can promise the list equals the citations the answer will carry. An event without it
 *     is a stage and a sentence, and nothing here invents a file list from one.
 *   - **The indicator clears the moment the answer starts.** It is a "nothing is on screen
 *     yet" indicator, not a progress bar: once a token has arrived the answer speaks for
 *     itself, and `done` and `error` clear it too.
 *
 * It lives here rather than in the component so that it can be tested; the component holds
 * the result in state and renders it, and adds no wording of its own.
 */

/**
 * One stage of answering, as the server reports it. `stage` is a stable name the UI can key
 * off; `content` is prose and may be reworded. `files` and `count` arrive only with the
 * "retrieved" stage, and only on a path where the server can promise they are the citations
 * the answer will carry - everywhere else the status is the stage and the sentence alone.
 *
 * Declared here rather than in lib/api.ts so that this module imports nothing: `npm test`
 * runs the TypeScript directly, and lib/api.ts pulls in the fetch wrapper with it.
 */
export interface StreamStatus {
    stage?: string;
    content: string;
    count?: number;
    files?: { path: string; start_line: number; end_line: number }[];
}

/** The stage sent the moment retrieval finishes, before the model has been asked anything. */
export const STAGE_RETRIEVED = "retrieved";

export interface ChatStatus {
    /** The stable stage name, or "" when an older server sent none. */
    stage: string;
    /** The sentence shown beside the spinner. */
    label: string;
    /** Distinct file paths, in the order the server sent them. Empty unless it sent files. */
    files: string[];
    /** "path (lines 10-42)" per location, for the tooltip. A file may appear twice. */
    locations: string[];
}

export function statusFromEvent(event: StreamStatus): ChatStatus {
    const sent = event.files ?? [];
    const files: string[] = [];
    for (const file of sent) {
        if (file?.path && !files.includes(file.path)) files.push(file.path);
    }
    return {
        stage: event.stage ?? "",
        label: event.content ?? "",
        files,
        locations: sent.map((f) => `${f.path} (lines ${f.start_line}-${f.end_line})`),
    };
}

/** Whether the waiting indicator is on screen: a status, and no answer text yet. */
export function isStatusVisible(status: ChatStatus | null, answer: string): boolean {
    return status !== null && answer === "";
}

/** True once the search has reported, which is when the file names are worth drawing. */
export function hasFoundFiles(status: ChatStatus | null): boolean {
    return status?.stage === STAGE_RETRIEVED && status.files.length > 0;
}

/** The last file name is joined with "and", so the line reads as a sentence rather than a list. */
export function fileSummary(status: ChatStatus | null, max = 4): string {
    const files = status?.files ?? [];
    if (files.length === 0) return "";
    const names = files.map((path) => path.split("/").pop() || path);
    if (names.length <= max) return names.join(", ");
    return `${names.slice(0, max).join(", ")} and ${names.length - max} more`;
}
