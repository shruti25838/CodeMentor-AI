"""Real code handed between agents, with the file and line range it came from.

Retrieval used to pass a prose summary to the next step, so the mentor answered from a
summary of code it had never seen and invented line numbers. The agents now pass these
instead, and every snippet carries the path and line range it was read from.

Two caps matter. One snippet is clipped so a single large file cannot fill the prompt, and
the set is clipped so the whole context stays well under the model's per-minute token
allowance. Before this, file-scope records were sent in full: one /ask against itsdangerous
asked for 10,391 tokens against an 8,000 token-per-minute limit and was rejected with 413.
"""

from dataclasses import dataclass

# Characters, not tokens, so no tokenizer dependency. About 4 characters per token, the
# same rough figure conversation memory uses.
DEFAULT_SNIPPET_MAX_CHARS = 1_500
DEFAULT_TOTAL_MAX_CHARS = 6_000

CLIPPED_MARKER = "... [clipped]"


@dataclass(frozen=True)
class Snippet:
    """A piece of a file, as read from disk. `text` may have been clipped."""

    path: str
    start_line: int
    end_line: int
    text: str
    clipped: bool = False

    @property
    def location(self) -> str:
        if self.start_line and self.end_line:
            return f"{self.path} (lines {self.start_line}-{self.end_line})"
        return self.path

    def render(self) -> str:
        """How a snippet is shown to a model: location, then the code."""
        return f"[{self.location}]\n{self.text}"


def clip_snippet(
    path: str,
    text: str,
    start_line: int,
    end_line: int,
    max_chars: int = DEFAULT_SNIPPET_MAX_CHARS,
) -> Snippet:
    """A snippet of at most max_chars, cut on a line boundary, with the line range it kept."""
    if len(text) <= max_chars:
        return Snippet(path=path, start_line=start_line, end_line=end_line, text=text, clipped=False)

    kept_lines: list[str] = []
    used = 0
    for line in text.splitlines():
        # +1 for the newline that joins it to the previous line.
        cost = len(line) + 1
        if used + cost > max_chars:
            break
        kept_lines.append(line)
        used += cost

    if not kept_lines:
        # A single line longer than the cap: cut mid-line rather than return nothing.
        return Snippet(
            path=path,
            start_line=start_line,
            end_line=start_line,
            text=text[:max_chars] + CLIPPED_MARKER,
            clipped=True,
        )

    return Snippet(
        path=path,
        start_line=start_line,
        end_line=start_line + len(kept_lines) - 1,
        text="\n".join(kept_lines) + "\n" + CLIPPED_MARKER,
        clipped=True,
    )


def fit_snippets(snippets: list[Snippet], max_chars: int = DEFAULT_TOTAL_MAX_CHARS) -> list[Snippet]:
    """As many whole snippets as fit, best first. A snippet that does not fit is dropped, not cut."""
    kept: list[Snippet] = []
    used = 0
    for snippet in snippets:
        cost = len(snippet.render())
        if used + cost > max_chars:
            continue
        kept.append(snippet)
        used += cost
    return kept


def render_snippets(snippets: list[Snippet]) -> str:
    return "\n\n".join(snippet.render() for snippet in snippets)
