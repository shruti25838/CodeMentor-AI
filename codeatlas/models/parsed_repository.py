from dataclasses import dataclass

from codeatlas.models.function_node import FunctionNode
from codeatlas.models.source_file import SourceFile


@dataclass(frozen=True)
class ParsedRepository:
    repository_id: str
    files: list[SourceFile]
    functions: list[FunctionNode]
