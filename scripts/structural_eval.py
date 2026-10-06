"""Measure the call graph against Python's own `ast` module.

`ast` is the independent ground truth: it is a different parser, written by someone else,
and the facts below are derived from it without consulting `call_graph.py` at all. The two
agree or they do not, and where they do not the reason is printed.

    python scripts/structural_eval.py                 # both splits, both repositories
    python scripts/structural_eval.py --split held-out
    python scripts/structural_eval.py --repo itsdangerous --show-disagreements

Repositories are pinned by commit and cached under .codeatlas/eval-repos/. Nothing here
calls a language model.
"""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from codeatlas.services.analysis.call_graph import CallGraph  # noqa: E402

QUESTIONS = ROOT / "eval" / "structural_questions.json"
CACHE = ROOT / ".codeatlas" / "eval-repos"

REPOS = {
    "itsdangerous": ("https://github.com/pallets/itsdangerous", "672971d66a2ef9f85151e53283113f33d642dabd"),
    "flask": ("https://github.com/pallets/flask", "d73fa1cdcbd8b1465c151db8924ba58b1dd14e35"),
}


# ---------------------------------------------------------------- ground truth (ast only)


def module_name(path: Path, root: Path) -> str:
    """Dotted module name from the path, dropping a leading src/ and a trailing __init__."""
    parts = list(path.relative_to(root).with_suffix("").parts)
    if parts and parts[0] == "src":
        parts = parts[1:]
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts) if parts else path.stem


def is_package(path: Path) -> bool:
    return path.name == "__init__.py"


@dataclass
class Truth:
    """Everything the evaluation needs, read with `ast`."""

    root: Path
    definitions: dict[str, set[str]]  # simple name -> {"path:line"}
    qualified: dict[str, set[str]]  # "Class.method" -> {"path:line"}
    call_sites: dict[str, set[str]]  # callee simple name -> {"path:line"}
    importers: dict[str, set[str]]  # module -> {importing module}
    modules: set[str]

    @classmethod
    def build(cls, root: Path) -> Truth:
        definitions: dict[str, set[str]] = {}
        qualified: dict[str, set[str]] = {}
        call_sites: dict[str, set[str]] = {}
        importers: dict[str, set[str]] = {}
        modules: set[str] = set()
        files = [p for p in sorted(root.rglob("*.py")) if ".git" not in p.parts]
        for path in files:
            modules.add(module_name(path, root))

        for path in files:
            rel = path.relative_to(root).as_posix()
            try:
                tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"), filename=str(path))
            except SyntaxError:
                continue
            me = module_name(path, root)
            package = me if is_package(path) else (me.rsplit(".", 1)[0] if "." in me else "")

            for node, stack in _walk_with_parents(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    definitions.setdefault(node.name, set()).add(f"{rel}:{node.lineno}")
                    if stack:
                        qualified.setdefault(f"{stack[-1]}.{node.name}", set()).add(f"{rel}:{node.lineno}")
                elif isinstance(node, ast.Call):
                    name = _callee_name(node.func)
                    if name:
                        call_sites.setdefault(name, set()).add(f"{rel}:{node.lineno}")
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        target = _resolve_absolute(alias.name, modules)
                        if target:
                            importers.setdefault(target, set()).add(me)
                elif isinstance(node, ast.ImportFrom):
                    target = _resolve_from(node, package, modules)
                    if target:
                        importers.setdefault(target, set()).add(me)

        return cls(root, definitions, qualified, call_sites, importers, modules)


def _walk_with_parents(tree: ast.AST):
    """Every node, with the stack of enclosing class names."""
    stack: list[str] = []

    def walk(node, stack):
        for child in ast.iter_child_nodes(node):
            yield child, list(stack)
            if isinstance(child, ast.ClassDef):
                yield from walk(child, stack + [child.name])
            else:
                yield from walk(child, stack)

    yield from walk(tree, stack)


def _callee_name(func: ast.AST) -> str:
    """The final name of a call target: foo() -> foo, a.b.foo() -> foo."""
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""


def _resolve_absolute(target: str, modules: set[str]) -> str:
    if target in modules:
        return target
    if "." not in target:
        return ""
    matches = sorted(m for m in modules if m.endswith(f".{target}"))
    return matches[0] if len(matches) == 1 else ""


def _resolve_from(node: ast.ImportFrom, package: str, modules: set[str]) -> str:
    if node.level:
        base = package
        for _ in range(node.level - 1):
            base = base.rsplit(".", 1)[0] if "." in base else ""
        candidate = f"{base}.{node.module}" if node.module and base else (node.module or base)
        return candidate if candidate in modules else ""
    return _resolve_absolute(node.module or "", modules)


# ---------------------------------------------------------------- scoring


@dataclass
class Scored:
    question: dict
    predicted: set[str]
    expected: set[str]

    @property
    def hits(self) -> set[str]:
        return self.predicted & self.expected

    @property
    def precision(self) -> float:
        return len(self.hits) / len(self.predicted) if self.predicted else (1.0 if not self.expected else 0.0)

    @property
    def recall(self) -> float:
        return len(self.hits) / len(self.expected) if self.expected else 1.0

    @property
    def exact(self) -> bool:
        return self.predicted == self.expected


def score(question: dict, graph: CallGraph, truth: Truth) -> Scored:
    kind, subject = question["kind"], question["subject"]

    if kind == "where_defined":
        predicted = {f"{d.path}:{d.start_line}" for d in graph.where_is_it_defined(subject)}
        expected = truth.qualified.get(subject) if "." in subject else truth.definitions.get(subject)
        return Scored(question, predicted, set(expected or ()))

    if kind == "who_calls":
        predicted = {f"{c.path}:{c.line}" for c in graph.who_calls(subject)}
        # Only call sites whose target is defined in this repository can be resolved, so the
        # ground truth is restricted the same way: a name with no definition here is excluded.
        expected = set(truth.call_sites.get(subject, set())) if subject in truth.definitions else set()
        return Scored(question, predicted, expected)

    if kind == "who_imports":
        predicted = {e.module for e in graph.who_imports(subject)}
        return Scored(question, predicted, set(truth.importers.get(subject, set())))

    raise ValueError(f"unknown question kind: {kind}")


# ---------------------------------------------------------------- repositories


def repo_path(name: str) -> Path:
    url, commit = REPOS[name]
    target = CACHE / f"{name}-{commit[:12]}"
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"cloning {name} at {commit[:12]} ...", flush=True)
    subprocess.run(["git", "init", "-q", str(target)], check=True)
    subprocess.run(["git", "-C", str(target), "remote", "add", "origin", url], check=True)
    subprocess.run(["git", "-C", str(target), "fetch", "-q", "--depth", "1", "origin", commit], check=True)
    subprocess.run(["git", "-C", str(target), "checkout", "-q", "FETCH_HEAD"], check=True)
    return target


# ---------------------------------------------------------------- report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["development", "held-out", "all"], default="all")
    parser.add_argument("--repo", choices=[*REPOS, "all"], default="all")
    parser.add_argument("--show-disagreements", action="store_true")
    args = parser.parse_args()

    questions = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    wanted = [q for q in questions if (args.split in ("all", q["split"])) and (args.repo in ("all", q["repo"]))]
    if not wanted:
        print("no questions selected")
        return 1

    started = time.perf_counter()
    graphs: dict[str, CallGraph] = {}
    truths: dict[str, Truth] = {}
    for name in sorted({q["repo"] for q in wanted}):
        path = repo_path(name)
        t = time.perf_counter()
        graphs[name] = CallGraph.build(path)
        build_s = time.perf_counter() - t
        t = time.perf_counter()
        truths[name] = Truth.build(path)
        truth_s = time.perf_counter() - t
        print(f"{name}: call graph {build_s:.2f}s, ast ground truth {truth_s:.2f}s")

    scored = [score(q, graphs[q["repo"]], truths[q["repo"]]) for q in wanted]
    elapsed = time.perf_counter() - started

    print()
    _table(scored)
    disagreements = [s for s in scored if not s.exact]
    print(f"\ntotal wall time {elapsed:.2f}s | {len(disagreements)} of {len(scored)} questions disagree with ast")

    if args.show_disagreements and disagreements:
        print("\nDisagreements")
        for s in disagreements:
            q = s.question
            print(f"\n  [{q['repo']}/{q['split']}] {q['kind']} {q['subject']}")
            missed = sorted(s.expected - s.predicted)
            extra = sorted(s.predicted - s.expected)
            if missed:
                print(f"    ast found, graph missed ({len(missed)}): {', '.join(missed[:6])}")
            if extra:
                print(f"    graph found, ast did not ({len(extra)}): {', '.join(extra[:6])}")
            print(f"    note: {q.get('note', '-')}")
    return 0


def _table(scored: list[Scored]) -> None:
    def summarise(rows: list[Scored]) -> str:
        if not rows:
            return "  (none)"
        p = sum(r.precision for r in rows) / len(rows)
        r_ = sum(r.recall for r in rows) / len(rows)
        exact = sum(1 for r in rows if r.exact)
        return f"  n={len(rows):3d}  precision={p:.3f}  recall={r_:.3f}  exact={exact}/{len(rows)}"

    print("by split")
    for split in ("development", "held-out"):
        print(f" {split:12s}{summarise([s for s in scored if s.question['split'] == split])}")
    print("by repository")
    for repo in sorted({s.question["repo"] for s in scored}):
        print(f" {repo:12s}{summarise([s for s in scored if s.question['repo'] == repo])}")
    print("by question kind")
    for kind in ("who_calls", "who_imports", "where_defined"):
        print(f" {kind:12s}{summarise([s for s in scored if s.question['kind'] == kind])}")
    print("overall")
    print(f" {'all':12s}{summarise(scored)}")


if __name__ == "__main__":
    raise SystemExit(main())
