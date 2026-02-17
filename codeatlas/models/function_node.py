from dataclasses import dataclass


@dataclass(frozen=True)
class FunctionNode:
    name: str
    file_path: str
    start_line: int
    end_line: int
    signature: str
