from pydantic import BaseModel


class ListFilesRequest(BaseModel):
    repo_id: str


class FileEntry(BaseModel):
    path: str
    language: str


class ListFilesResponse(BaseModel):
    files: list[FileEntry]


class FileContentRequest(BaseModel):
    repo_id: str
    file_path: str


class FileContentResponse(BaseModel):
    path: str
    content: str
    language: str
    line_count: int
