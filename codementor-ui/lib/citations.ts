/** A citation from the backend: "path (lines 3-10) | snippet", "path | snippet" or just "path". */
export interface ParsedCitation {
    path: string;
    startLine?: number;
    endLine?: number;
    snippet: string;
}

const LINE_RANGE = /\s*\(lines (\d+)-(\d+)\)$/;

export function parseCitation(citation: string): ParsedCitation {
    const sep = citation.indexOf(" | ");
    const head = (sep === -1 ? citation : citation.slice(0, sep)).trim();
    const snippet = sep === -1 ? "" : citation.slice(sep + 3).trim();
    const range = head.match(LINE_RANGE);
    if (!range) return { path: head, snippet };
    const startLine = Number(range[1]);
    const endLine = Number(range[2]);
    return {
        path: head.slice(0, range.index).trim(),
        startLine,
        endLine: endLine >= startLine ? endLine : startLine,
        snippet,
    };
}

export interface FilePreviewRequest {
    path: string;
    startLine?: number;
    endLine?: number;
}

/** Ask the workspace to open the file preview (handled in app/workspace/page.tsx). */
export function openFilePreview(request: FilePreviewRequest) {
    window.dispatchEvent(new CustomEvent("codeatlas:preview-file", { detail: request }));
}
