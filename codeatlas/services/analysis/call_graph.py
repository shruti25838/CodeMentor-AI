"""A call and import graph for Python, built with Tree-sitter. No model is involved.

Every fact this module reports is read from the syntax tree. The analyst agent may ask a
model to phrase an answer, but never to produce one of these facts.

Resolution is deliberately conservative. A call site is resolved to a definition in this
repository only when the syntax plus the file's imports make the target unambiguous; every
other call is kept and marked unresolved rather than guessed at. `WHAT_IT_MISSES` lists the
cases that stay unresolved, and the structural evaluation measures how often that happens.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from tree_sitter import Node
from tree_sitter_languages import get_parser

logger = logging.getLogger(__name__)

# Said plainly here and repeated in docs/RESULTS.md, because the honest limits of this
# graph matter more than its coverage number.
WHAT_IT_MISSES = """\
- Dynamic calls: getattr(obj, name)(), eval, dispatch tables, callbacks stored in variables.
- A call through a variable whose type is not obvious, e.g. `handler = pick(); handler()`.
- Decorators change what a name refers to at run time; the graph records the decorated
  function's own body and resolves calls to the undecorated definition.
- Inheritance is followed only when the base class is a plain name defined or imported in
  the same repository. Bases that are subscripted, aliased through several modules, or
  built at run time are not followed.
- `from x import *` is recorded as an import but contributes no names for resolution.
- Conditional and function-local imports are recorded but are not scoped to the branch.
- Attribute calls on a value returned by another call, e.g. `get_signer().sign()`.
- Only Python. Other languages in the repository are ignored entirely.
- Two definitions with the same qualified name (a name redefined in a branch) keep the
  first one seen."""


@dataclass(frozen=True)
class Definition:
    qualname: str
    name: str
    kind: str  # "function", "method" or "class"
    module: str
    path: str
    start_line: int
    end_line: int
    parent: str = ""  # enclosing class qualname, for methods

    @property
    def location(self) -> str:
        return f"{self.path}:{self.start_line}"


@dataclass(frozen=True)
class ImportEdge:
    module: str  # the module doing the importing
    path: str
    target: str  # module named in the statement, as written
    name: str  # symbol for `from x import name`; "" for `import x`
    alias: str
    line: int
    resolved_module: str = ""  # the repo module it refers to, or "" if outside the repo

    @property
    def internal(self) -> bool:
        return bool(self.resolved_module)


@dataclass(frozen=True)
class CallSite:
    caller: str  # qualname of the enclosing definition, or the module for top-level code
    callee_text: str  # the expression as written, e.g. "self.helper"
    callee: str  # resolved definition qualname, or "" when unresolved
    path: str
    line: int

    @property
    def resolved(self) -> bool:
        return bool(self.callee)


@dataclass
class _ModuleFacts:
    module: str
    path: str
    is_package: bool = False
    definitions: list[Definition] = field(default_factory=list)
    imports: list[ImportEdge] = field(default_factory=list)
    calls: list[CallSite] = field(default_factory=list)
    # name as used in this module -> what it refers to
    imported_names: dict[str, str] = field(default_factory=dict)  # name -> "module:symbol"
    imported_modules: dict[str, str] = field(default_factory=dict)  # alias -> module
    base_classes: dict[str, list[str]] = field(default_factory=dict)  # class qualname -> base names


class CallGraph:
    """Definitions, imports and call sites for the Python files under a root."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.definitions: dict[str, Definition] = {}
        self.by_name: dict[str, list[Definition]] = defaultdict(list)
        self.imports: list[ImportEdge] = []
        self.calls: list[CallSite] = []
        self.modules: dict[str, _ModuleFacts] = {}
        self.parse_errors: list[str] = []

    # ---------- building ----------

    @classmethod
    def build(cls, root: str | Path, max_files: int | None = None) -> CallGraph:
        graph = cls(root)
        paths = sorted(p for p in graph.root.rglob("*.py") if ".git" not in p.parts)
        if max_files is not None:
            paths = paths[:max_files]
        for path in paths:
            graph._collect(path)
        graph._index()
        graph._resolve()
        return graph

    def _collect(self, path: Path) -> None:
        try:
            source = path.read_bytes()
        except OSError as exc:
            self.parse_errors.append(f"{path}: {exc}")
            return
        try:
            tree = get_parser("python").parse(source)
        except Exception as exc:  # pragma: no cover - parser unavailable
            self.parse_errors.append(f"{path}: {exc}")
            return

        module = self._module_name(path)
        rel = path.relative_to(self.root).as_posix()
        facts = _ModuleFacts(module=module, path=rel, is_package=path.name == "__init__.py")
        _walk_module(tree.root_node, source, facts)
        self.modules[module] = facts

    def _module_name(self, path: Path) -> str:
        """Dotted module name from the path relative to the repository root.

        Walking up only while `__init__.py` exists is wrong for a namespace package:
        flask's `src/flask/sansio/` has no `__init__.py`, so that rule named
        `src/flask/sansio/app.py` just `app` and collided it with every other app.py.
        A leading `src/` is dropped because it is a packaging convention, not part of the
        importable name.
        """
        parts = list(path.relative_to(self.root).with_suffix("").parts)
        if parts and parts[0] == "src":
            parts = parts[1:]
        if parts and parts[-1] == "__init__":
            parts = parts[:-1]
        return ".".join(parts) if parts else path.stem

    def _index(self) -> None:
        self._known: set[str] = set(self.modules)
        for facts in self.modules.values():
            for definition in facts.definitions:
                # A name redefined later keeps the first definition seen.
                self.definitions.setdefault(definition.qualname, definition)
                self.by_name[definition.name].append(definition)
            self.imports.extend(facts.imports)

    def _package_of(self, module: str) -> str:
        """The package a relative import inside `module` is relative to.

        For a package's own __init__.py the module name *is* the package, so
        `from .signer import Signer` in itsdangerous/__init__.py means itsdangerous.signer.
        """
        facts = self.modules.get(module)
        if facts is not None and facts.is_package:
            return module
        return module.rsplit(".", 1)[0] if "." in module else ""

    # ---------- resolution ----------

    def _resolve(self) -> None:
        known = set(self.modules)
        resolved_imports: list[ImportEdge] = []
        for edge in self.imports:
            resolved_imports.append(
                ImportEdge(
                    module=edge.module,
                    path=edge.path,
                    target=edge.target,
                    name=edge.name,
                    alias=edge.alias,
                    line=edge.line,
                    resolved_module=_match_module(edge.target, known, edge.module, self._package_of(edge.module)),
                )
            )
        self.imports = resolved_imports
        for facts in self.modules.values():
            facts.imports = [e for e in resolved_imports if e.module == facts.module]

        for facts in self.modules.values():
            for call in facts.calls:
                facts_calls = self._resolve_call(call, facts, known)
                self.calls.append(facts_calls)

    def _resolve_call(self, call: CallSite, facts: _ModuleFacts, known: set[str]) -> CallSite:
        target = self._lookup(call.callee_text, call.caller, facts, known)
        return CallSite(
            caller=call.caller,
            callee_text=call.callee_text,
            callee=target,
            path=call.path,
            line=call.line,
        )

    def _lookup(self, text: str, caller: str, facts: _ModuleFacts, known: set[str]) -> str:
        if not text:
            return ""
        parts = text.split(".")

        if len(parts) == 1:
            return self._lookup_bare(parts[0], caller, facts)

        head, attr = ".".join(parts[:-1]), parts[-1]

        # self.method() / cls.method(): look in the enclosing class, then its bases.
        if head in ("self", "cls"):
            owner = _enclosing_class(caller, facts)
            return self._method_on(owner, attr, facts) if owner else ""

        # module.func(), where the module was imported in this file.
        module = facts.imported_modules.get(head)
        if module:
            resolved = _match_module(module, known, facts.module, self._package_of(facts.module))
            if resolved:
                return self._in_module(resolved, attr)
            return ""

        # Class.method(), where Class is defined here or imported from a repo module.
        owner = self._class_named(head, facts)
        if owner:
            return self._method_on(owner, attr, facts)

        return ""

    def _lookup_bare(self, name: str, caller: str, facts: _ModuleFacts) -> str:
        # A nested definition inside the calling function wins.
        nested = f"{caller}.{name}"
        if nested in self.definitions:
            return nested
        # Then a module-level definition in the same module.
        local = f"{facts.module}.{name}"
        if local in self.definitions:
            return local
        # Then a name brought in by `from module import name`.
        imported = facts.imported_names.get(name)
        if imported:
            module, _, symbol = imported.partition(":")
            return self._in_module(module, symbol, facts.module)
        return ""

    def _in_module(self, module: str, symbol: str, importer: str = "") -> str:
        # `from .encoding import want_bytes` records ".encoding"; map it to the real module.
        known = getattr(self, "_known", set(self.modules))
        module = _match_module(module, known, importer, self._package_of(importer) if importer else None) or module
        qualname = f"{module}.{symbol}"
        if qualname in self.definitions:
            return qualname
        # The module may re-export the name; follow one hop of `from x import symbol`.
        facts = self.modules.get(module)
        if facts:
            onward = facts.imported_names.get(symbol)
            if onward:
                next_module, _, next_symbol = onward.partition(":")
                if next_module != module:
                    return self._in_module(next_module, next_symbol, module)
        return ""

    def _class_named(self, name: str, facts: _ModuleFacts) -> str:
        local = f"{facts.module}.{name}"
        if local in self.definitions and self.definitions[local].kind == "class":
            return local
        imported = facts.imported_names.get(name)
        if imported:
            module, _, symbol = imported.partition(":")
            qualname = self._in_module(module, symbol, facts.module)
            if qualname and self.definitions[qualname].kind == "class":
                return qualname
        return ""

    def _method_on(self, class_qualname: str, method: str, facts: _ModuleFacts) -> str:
        """A method on a class or, failing that, on a base class defined in this repo."""
        direct = f"{class_qualname}.{method}"
        if direct in self.definitions:
            return direct
        owner_module = self.definitions[class_qualname].module if class_qualname in self.definitions else facts.module
        owner_facts = self.modules.get(owner_module, facts)
        seen = {class_qualname}
        queue = list(owner_facts.base_classes.get(class_qualname, []))
        while queue:
            base_name = queue.pop(0)
            base = self._class_named(base_name, owner_facts)
            if not base or base in seen:
                continue
            seen.add(base)
            candidate = f"{base}.{method}"
            if candidate in self.definitions:
                return candidate
            base_module = self.definitions[base].module
            queue.extend(self.modules.get(base_module, owner_facts).base_classes.get(base, []))
        return ""

    # ---------- questions ----------

    def who_calls(self, name: str) -> list[CallSite]:
        """Call sites whose resolved target is `name` (a qualified name or a plain one)."""
        targets = self._targets(name)
        return [c for c in self.calls if c.callee in targets]

    def what_does_it_call(self, name: str) -> list[CallSite]:
        """Calls made from inside `name`, including the unresolved ones."""
        targets = self._targets(name)
        return [c for c in self.calls if c.caller in targets]

    def where_is_it_defined(self, name: str) -> list[Definition]:
        return sorted(self._definitions_for(name), key=lambda d: (d.path, d.start_line))

    def who_imports(self, module_or_name: str) -> list[ImportEdge]:
        """Import statements that refer to `module_or_name`, by module or by symbol."""
        wanted = module_or_name.strip()
        return [edge for edge in self.imports if _refers_to(edge, wanted)]

    def what_does_it_import(self, module: str) -> list[ImportEdge]:
        facts = self.modules.get(module) or self._module_by_suffix(module)
        return list(facts.imports) if facts else []

    def import_edges(self) -> dict[str, set[str]]:
        """Module -> modules it imports, counting only modules inside this repository."""
        edges: dict[str, set[str]] = defaultdict(set)
        for edge in self.imports:
            if edge.internal and edge.resolved_module != edge.module:
                edges[edge.module].add(edge.resolved_module)
        return edges

    def circular_imports(self) -> list[list[str]]:
        """Groups of modules that import each other, directly or through other modules.

        Each group is a strongly connected component: every module in it can reach every
        other. Groups are returned, not every distinct loop, because the number of distinct
        loops grows exponentially — flask has 33,796 of them and listing those says nothing
        a reader can act on, while the groups say exactly which modules are tangled.
        """
        edges = self.import_edges()
        nodes = sorted(set(edges) | {t for targets in edges.values() for t in targets})
        groups = [sorted(c) for c in _strongly_connected(nodes, edges) if len(c) > 1]
        # A module importing itself is a cycle of one.
        groups += [[n] for n in nodes if n in edges.get(n, ())]
        return sorted(groups)

    def shortest_import_cycle(self, module: str) -> list[str]:
        """The shortest import loop starting and ending at `module`, or [] if there is none."""
        edges = self.import_edges()
        if module not in edges:
            return []
        queue: list[list[str]] = [[module]]
        seen = {module}
        while queue:
            path = queue.pop(0)
            for nxt in sorted(edges.get(path[-1], ())):
                if nxt == module:
                    return path + [module]
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append(path + [nxt])
        return []

    # ---------- counts ----------

    def resolution_rate(self) -> tuple[int, int]:
        resolved = sum(1 for c in self.calls if c.resolved)
        return resolved, len(self.calls)

    def _targets(self, name: str) -> set[str]:
        return {d.qualname for d in self._definitions_for(name)}

    def _definitions_for(self, name: str) -> list[Definition]:
        name = name.strip()
        if name in self.definitions:
            return [self.definitions[name]]
        exact = self.by_name.get(name, [])
        if exact:
            return list(exact)
        # Allow a partly-qualified name such as "Signer.sign".
        return [d for q, d in self.definitions.items() if q.endswith(f".{name}")]

    def _module_by_suffix(self, module: str) -> _ModuleFacts | None:
        for name, facts in self.modules.items():
            if name == module or name.endswith(f".{module}"):
                return facts
        return None


# ---------- syntax walking ----------


def _text(node: Node, source: bytes) -> str:
    return source[node.start_byte : node.end_byte].decode("utf-8", errors="replace")


def _walk_module(root: Node, source: bytes, facts: _ModuleFacts) -> None:
    for child in root.children:
        _visit(child, source, facts, scope=facts.module, class_scope="")


def _visit(node: Node, source: bytes, facts: _ModuleFacts, scope: str, class_scope: str) -> None:
    """One pass over the tree. Each node is seen once, so each call is recorded once."""
    kind = node.type

    if kind == "function_definition":
        name_node = node.child_by_field_name("name")
        if name_node is None:
            return
        name = _text(name_node, source)
        qualname = f"{scope}.{name}"
        facts.definitions.append(
            Definition(
                qualname=qualname,
                name=name,
                kind="method" if class_scope else "function",
                module=facts.module,
                path=facts.path,
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                parent=class_scope,
            )
        )
        body = node.child_by_field_name("body")
        # child_by_field_name returns a fresh wrapper for the same node, so compare ids:
        # `child is body` is never true and would walk the body in the enclosing scope.
        body_id = body.id if body is not None else None
        for child in node.children:
            if child.id == body_id:
                # The body belongs to the function's own scope.
                for stmt in child.children:
                    _visit(stmt, source, facts, scope=qualname, class_scope="")
            else:
                # Default arguments and return annotations run in the enclosing scope.
                _visit(child, source, facts, scope=scope, class_scope=class_scope)
        return

    if kind == "class_definition":
        name_node = node.child_by_field_name("name")
        if name_node is None:
            return
        name = _text(name_node, source)
        qualname = f"{scope}.{name}"
        facts.definitions.append(
            Definition(
                qualname=qualname,
                name=name,
                kind="class",
                module=facts.module,
                path=facts.path,
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                parent=class_scope,
            )
        )
        facts.base_classes[qualname] = _base_names(node, source)
        body = node.child_by_field_name("body")
        if body is not None:
            for stmt in body.children:
                _visit(stmt, source, facts, scope=qualname, class_scope=qualname)
        return

    if kind == "import_statement":
        _record_import(node, source, facts)
        return

    if kind == "import_from_statement":
        _record_from_import(node, source, facts)
        return

    if kind == "call":
        function_node = node.child_by_field_name("function")
        if function_node is not None and function_node.type in ("identifier", "attribute"):
            facts.calls.append(
                CallSite(
                    caller=scope,
                    callee_text=_text(function_node, source),
                    callee="",
                    path=facts.path,
                    line=node.start_point[0] + 1,
                )
            )
        # Arguments may hold further calls, e.g. f(g()).
        for child in node.children:
            _visit(child, source, facts, scope=scope, class_scope=class_scope)
        return

    for child in node.children:
        _visit(child, source, facts, scope=scope, class_scope=class_scope)


def _base_names(class_node: Node, source: bytes) -> list[str]:
    args = class_node.child_by_field_name("superclasses")
    if args is None:
        return []
    names = []
    for child in args.children:
        if child.type in ("identifier", "attribute"):
            names.append(_text(child, source))
    return names


def _record_import(node: Node, source: bytes, facts: _ModuleFacts) -> None:
    line = node.start_point[0] + 1
    for child in node.children:
        if child.type == "dotted_name":
            target = _text(child, source)
            facts.imports.append(ImportEdge(facts.module, facts.path, target, "", "", line))
            facts.imported_modules[target.split(".")[0]] = target.split(".")[0]
            facts.imported_modules[target] = target
        elif child.type == "aliased_import":
            name_node = child.child_by_field_name("name")
            alias_node = child.child_by_field_name("alias")
            if name_node is None or alias_node is None:
                continue
            target = _text(name_node, source)
            alias = _text(alias_node, source)
            facts.imports.append(ImportEdge(facts.module, facts.path, target, "", alias, line))
            facts.imported_modules[alias] = target


def _record_from_import(node: Node, source: bytes, facts: _ModuleFacts) -> None:
    line = node.start_point[0] + 1
    module_node = node.child_by_field_name("module_name")
    target = _text(module_node, source) if module_node is not None else ""

    wildcard = any(c.type == "wildcard_import" for c in node.children)
    if wildcard:
        facts.imports.append(ImportEdge(facts.module, facts.path, target, "*", "", line))
        return

    for child in node.children:
        if child is module_node:
            continue
        if child.type == "dotted_name":
            name = _text(child, source)
            facts.imports.append(ImportEdge(facts.module, facts.path, target, name, "", line))
            facts.imported_names[name] = f"{target}:{name}"
            facts.imported_modules.setdefault(name, f"{target}.{name}")
        elif child.type == "aliased_import":
            name_node = child.child_by_field_name("name")
            alias_node = child.child_by_field_name("alias")
            if name_node is None or alias_node is None:
                continue
            name = _text(name_node, source)
            alias = _text(alias_node, source)
            facts.imports.append(ImportEdge(facts.module, facts.path, target, name, alias, line))
            facts.imported_names[alias] = f"{target}:{name}"
            facts.imported_modules.setdefault(alias, f"{target}.{name}")


def _enclosing_class(caller: str, facts: _ModuleFacts) -> str:
    """The class a method belongs to, from its qualified name."""
    parent = caller.rsplit(".", 1)[0] if "." in caller else ""
    while parent:
        if parent in facts.base_classes:
            return parent
        if "." not in parent:
            return ""
        parent = parent.rsplit(".", 1)[0]
    return ""


def _match_module(target: str, known: set[str], importer: str = "", package: str | None = None) -> str:
    """The repo module a written import refers to, or "" when it is outside the repo.

    A relative import is resolved against the importing module's own package, which is the
    only correct reading: `from .encoding import x` inside `itsdangerous.timed` means
    `itsdangerous.encoding`, never some other `encoding` elsewhere in the tree.
    """
    if not target:
        return ""

    if target.startswith("."):
        if not importer:
            return ""
        # One dot means this package; each extra dot goes one package further up.
        up = len(target) - len(target.lstrip("."))
        rest = target.lstrip(".")
        if package is None:
            package = importer.rsplit(".", 1)[0] if "." in importer else ""
        for _ in range(up - 1):
            package = package.rsplit(".", 1)[0] if "." in package else ""
        candidate = f"{package}.{rest}" if package and rest else (rest or package)
        return candidate if candidate in known else ""

    if target in known:
        return target

    # A dotted absolute import may still name a repo module under a different prefix. A
    # single-segment target is never matched this way: `import logging` must stay the
    # standard library, not become this repo's `flask.logging`.
    if "." not in target:
        return ""
    matches = sorted(c for c in known if c.endswith(f".{target}"))
    if len(matches) == 1:
        return matches[0]
    if matches and importer:
        # Ambiguous: prefer the candidate sharing the longest package prefix with the importer.
        def shared(candidate: str) -> int:
            a, b = candidate.split("."), importer.split(".")
            n = 0
            while n < min(len(a), len(b)) and a[n] == b[n]:
                n += 1
            return n

        best = max(matches, key=shared)
        return best if shared(best) else ""
    return ""


def _refers_to(edge: ImportEdge, wanted: str) -> bool:
    """Whether an import statement refers to a module or an imported symbol by that name."""
    if edge.resolved_module == wanted or edge.target == wanted:
        return True
    if edge.name == wanted and edge.internal:
        return True
    return bool(edge.resolved_module) and edge.resolved_module.endswith(f".{wanted}")


def _strongly_connected(nodes: list[str], edges: dict[str, set[str]]) -> list[list[str]]:
    """Tarjan's algorithm, iterative so a deep import chain cannot exhaust the stack."""
    index: dict[str, int] = {}
    low: dict[str, int] = {}
    on_stack: set[str] = set()
    stack: list[str] = []
    result: list[list[str]] = []
    counter = 0

    for root in nodes:
        if root in index:
            continue
        work: list[tuple[str, int]] = [(root, 0)]
        while work:
            node, child_i = work[-1]
            if child_i == 0:
                index[node] = low[node] = counter
                counter += 1
                stack.append(node)
                on_stack.add(node)
            children = sorted(edges.get(node, ()))
            if child_i < len(children):
                work[-1] = (node, child_i + 1)
                child = children[child_i]
                if child not in index:
                    work.append((child, 0))
                elif child in on_stack:
                    low[node] = min(low[node], index[child])
                continue
            if low[node] == index[node]:
                component = []
                while True:
                    member = stack.pop()
                    on_stack.discard(member)
                    component.append(member)
                    if member == node:
                        break
                result.append(component)
            work.pop()
            if work:
                parent = work[-1][0]
                low[parent] = min(low[parent], low[node])
    return result
