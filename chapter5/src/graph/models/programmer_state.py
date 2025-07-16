from typing import TypedDict

from pydantic import BaseModel, Field

from src.models import SubTask


class DataThread(BaseModel):
    user_request: str | None
    code: str | None = None
    error: str | None = None
    stderr: str | None = None
    stdout: str | None = None
    is_completed: bool = False
    observation: str | None = None
    results: list[dict] = Field(default_factory=list)
    pathes: dict = Field(default_factory=dict)


class ProgrammerState(TypedDict):
    data_file: str
    data_info: str
    user_request: str
    data_threads: list[DataThread]
    sandbox_id: str
    next_node: str
    sub_tasks: list[SubTask]
