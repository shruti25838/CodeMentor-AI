from abc import ABC, abstractmethod

from codeatlas.models.parsed_repository import ParsedRepository
from codeatlas.models.repository import Repository


class AstParser(ABC):
    @abstractmethod
    def parse_repository(self, repository: Repository) -> ParsedRepository:
        raise NotImplementedError
