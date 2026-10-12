"""What a record *is*: the whole file and the whole function, or something shaped for retrieval.

Today's format, `whole`, makes one record per file and one per function, each holding that
thing's entire text. Two problems follow from it, and both show up in `docs/RESULTS.md`:

* **A file record is mostly not about the file.** flask's `app.py` is 1,628 lines; embedded as
  one vector it is an average of everything in it, and under a clip it is whatever happens to
  be in the first 2,000 characters - which for a Python module is the licence header and the
  imports. 54.2% of flask's file records are over the clip and keep 18.6% of their characters.
* **A long function is embedded by its opening.** The same clip cuts 52 of flask's function
  records, and a window of the body that answers the question may be nowhere near the start.

`structured` addresses each directly:

* a **file record becomes a header** - the path, the module docstring, the imports, and the
  signature of every class and function in the file, capped near the clip. That is a summary
  of what the file is *for*, and it fits, so the clip stops throwing away the part that
  matters.
* a **function longer than the window becomes several overlapping windows**, each its own
  record with its own accurate line range. The overlap is there so that a passage straddling
  a boundary is whole in one of them.

Windows are cut **on line boundaries**, never mid-line, because a record's line range has to be
true: it is what the citation shows and what the file preview opens. A single line longer than
the window is its own window rather than being split.

Because several records can now point into one function, ranking deduplicates them: see
`AnswerService._dedupe_windows`. The visitor sees one result per function, the best-scoring
window of it.

This changes every vector in an index, so the format is part of the embedder's signature (see
`with_record_format`) and therefore part of the index format and the mismatch guard. An index
built under one format is refused by a server running the other, rather than searched and
quietly answered wrongly.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from codeatlas.services.retrieval.embedding import EmbeddingService

WHOLE = "whole"
STRUCTURED = "structured"
RECORD_FORMATS = (WHOLE, STRUCTURED)

# Used when no per-document clip is set, which is the case for the default (hash) provider.
DEFAULT_WINDOW_CHARS = 2000
DEFAULT_WINDOW_OVERLAP = 200

# Lines that introduce a dependency, in the languages this repository parses. Matched at the
# start of a line so that the word inside a string or a comment is not mistaken for one.
_IMPORT = re.compile(r"^\s*(?:import|from|#include|using|require|package|use|extern crate)\b")
_DOCSTRING = re.compile(r'^\s*(?P<q>"""|\'\'\')')


def unknown_record_format_message(value: str) -> str:
    return (
        f"Unknown record format {value!r}. Set CODEATLAS_RECORD_FORMAT to one of: "
        f"{', '.join(RECORD_FORMATS)}. '{WHOLE}' is the default and is what every number in "
        f"docs/RESULTS.md was measured with."
    )


def validate_record_format(value: str) -> str:
    normalized = value.strip().lower()
    if normalized not in RECORD_FORMATS:
        raise ValueError(unknown_record_format_message(value))
    return normalized


@dataclass(frozen=True)
class Window:
    """One slice of a record's text, with the line range it really covers (1-based, inclusive)."""

    text: str
    start_line: int
    end_line: int


def windows(text: str, start_line: int, window_chars: int, overlap_chars: int) -> list[Window]:
    """`text` cut into overlapping windows on line boundaries, each at most `window_chars`.

    `start_line` is the line the text begins on in its file, so every window's range is a real
    range in that file. Text that already fits comes back as one window, unchanged, which is
    what keeps a short function identical to what `whole` produced.
    """
    if window_chars <= 0:
        raise ValueError("window_chars must be positive")
    lines = text.splitlines()
    if not lines:
        return [Window(text=text, start_line=start_line, end_line=start_line)]
    if len(text) <= window_chars:
        return [Window(text=text, start_line=start_line, end_line=start_line + len(lines) - 1)]

    cut: list[Window] = []
    index = 0
    while index < len(lines):
        end = index
        size = 0
        while end < len(lines) and size + len(lines[end]) + 1 <= window_chars:
            size += len(lines[end]) + 1
            end += 1
        # One line longer than the whole window: it becomes a window of its own rather than
        # being split, because half a line is not a thing anyone can read or cite.
        if end == index:
            end = index + 1
        cut.append(
            Window(
                text="\n".join(lines[index:end]),
                start_line=start_line + index,
                end_line=start_line + end - 1,
            )
        )
        if end >= len(lines):
            break
        # Step back far enough to repeat about `overlap_chars` of the tail, but always leave
        # at least one new line ahead, so the walk cannot stall.
        back = 0
        step = end
        while step > index + 1 and back + len(lines[step - 1]) + 1 <= overlap_chars:
            step -= 1
            back += len(lines[step]) + 1
        index = step
    return cut


def file_header(text: str, label: str, signatures: list[str], max_chars: int) -> str:
    """What a file record holds under `structured`: what the file is, not everything in it.

    The path, the module docstring, the import lines and the signature of every class and
    function, in that order, cut to `max_chars`. The order is deliberate: the path and the
    docstring say what the file is for in the fewest characters, so they survive the cut.
    """
    parts: list[str] = [label] if label else []
    docstring = _module_docstring(text)
    if docstring:
        parts.append(docstring)
    imports = [line.strip() for line in text.splitlines() if _IMPORT.match(line)]
    if imports:
        parts.append("\n".join(dict.fromkeys(imports)))
    if signatures:
        parts.append("\n".join(signatures))
    header = "\n".join(parts)
    return header[:max_chars] if max_chars > 0 else header


def _module_docstring(text: str) -> str:
    """The leading triple-quoted string, if the file opens with one, blank lines and comments aside."""
    lines = text.splitlines()
    index = 0
    while index < len(lines) and (not lines[index].strip() or lines[index].lstrip().startswith("#")):
        index += 1
    if index >= len(lines):
        return ""
    opening = _DOCSTRING.match(lines[index])
    if not opening:
        return ""
    quote = opening.group("q")
    rest = lines[index].strip()[3:]
    if rest.endswith(quote) and len(rest) >= 3:
        return lines[index].strip()
    collected = [lines[index].strip()]
    for line in lines[index + 1 :]:
        collected.append(line.rstrip())
        if quote in line:
            break
    return "\n".join(collected)


class RecordFormatEmbedder(EmbeddingService):
    """An embedder whose signature also names the record format the index was built with.

    Vectors are the embedder's, unchanged. What changes is the *text* it is given, and that is
    exactly what a signature exists to pin down: `embedder_mismatch` compares the signature
    stored with an index against the one the server is running, so with the format in it an
    index built under `whole` is refused rather than searched by a `structured` server.

    Only ever wrapped around a non-default format, so `whole` keeps the signature it has always
    had and no existing index is invalidated.
    """

    def __init__(self, inner: EmbeddingService, record_format: str) -> None:
        self._inner = inner
        self._record_format = record_format

    @property
    def inner(self) -> EmbeddingService:
        return self._inner

    def signature(self) -> str:
        return f"{self._inner.signature()}|records={self._record_format}"

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return self._inner.embed_texts(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._inner.embed_query(text)

    def known(self, texts: list[str]) -> list[bool]:
        """Which of these the wrapped embedder has already paid for, when it keeps a cache.

        Passed through so that costing a run does not have to know whether a record format
        wrapper is in the way. An embedder with no cache has paid for nothing.
        """
        inner_known = getattr(self._inner, "known", None)
        return inner_known(texts) if inner_known else [False] * len(texts)


def with_record_format(embedder: EmbeddingService, record_format: str) -> EmbeddingService:
    """The embedder, with the record format in its signature unless the format is the default."""
    if record_format == WHOLE:
        return embedder
    return RecordFormatEmbedder(embedder, validate_record_format(record_format))
