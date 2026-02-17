from abc import ABC, abstractmethod

import networkx as nx

from codeatlas.models.parsed_repository import ParsedRepository


class DependencyGraphBuilder(ABC):
    @abstractmethod
    def build_import_graph(self, parsed_repo: ParsedRepository) -> nx.DiGraph:
        raise NotImplementedError
