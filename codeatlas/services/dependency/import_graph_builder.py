import logging
from pathlib import Path

import networkx as nx
from tree_sitter import Node
from tree_sitter_languages import get_parser

from codeatlas.models.parsed_repository import ParsedRepository
from codeatlas.services.dependency.interfaces import DependencyGraphBuilder


class ImportGraphBuilder(DependencyGraphBuilder):
    def build_import_graph(self, parsed_repo: ParsedRepository) -> nx.DiGraph:
        graph = nx.DiGraph()
        for source_file in parsed_repo.files:
            graph.add_node(source_file.path)
            language = self._language_from_suffix(Path(source_file.path).suffix)
            if language is None:
                continue
            targets = self._extract_imports(Path(source_file.path), language)
            for target in targets:
                graph.add_node(target)
                graph.add_edge(source_file.path, target, relation="imports")
        return graph

    def _language_from_suffix(self, suffix: str) -> str | None:
        if suffix == ".py":
            return "python"
        if suffix in {".js", ".jsx", ".ts", ".tsx"}:
            return "javascript"
        return None

    def _extract_imports(self, path: Path, language: str) -> list[str]:
        try:
            parser = get_parser(language)
        except Exception as exc:
            logging.warning("Tree-sitter parser unavailable for %s: %s", language, exc)
            return []

        try:
            source_bytes = path.read_bytes()
        except OSError as exc:
            logging.warning("Failed to read %s: %s", path, exc)
            return []

        tree = parser.parse(source_bytes)
        root = tree.root_node
        if language == "python":
            return self._collect_python_imports(root, source_bytes)
        return self._collect_js_imports(root, source_bytes)

    def _collect_python_imports(self, root: Node, source: bytes) -> list[str]:
        imports: set[str] = set()
        stack = [root]
        while stack:
            node = stack.pop()
            if node.type == "import_statement":
                for child in node.children:
                    if child.type == "dotted_name":
                        imports.add(self._node_text(child, source).strip())
            elif node.type == "import_from_statement":
                module_node = node.child_by_field_name("module_name")
                if module_node is not None:
                    imports.add(self._node_text(module_node, source).strip())
            stack.extend(reversed(node.children))
        return sorted(imports)

    def _collect_js_imports(self, root: Node, source: bytes) -> list[str]:
        imports: set[str] = set()
        stack = [root]
        while stack:
            node = stack.pop()
            if node.type in {"import_statement", "export_statement"}:
                source_node = node.child_by_field_name("source")
                if source_node is not None:
                    imports.add(self._strip_quotes(self._node_text(source_node, source)))
            stack.extend(reversed(node.children))
        return sorted(imports)

    def _strip_quotes(self, value: str) -> str:
        value = value.strip()
        if len(value) >= 2 and value[0] in {"'", '"'} and value[-1] == value[0]:
            return value[1:-1]
        return value

    def _node_text(self, node: Node, source: bytes) -> str:
        return source[node.start_byte : node.end_byte].decode(
            "utf-8", errors="replace"
        )
