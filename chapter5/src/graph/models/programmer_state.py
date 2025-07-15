from typing import TypedDict

from e2b_code_interpreter import Sandbox
from pydantic import BaseModel, Field


class DataThread(BaseModel):
    task_request: str | None
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
    sandbox: Sandbox
    current_thread_id: int
    process_id: str
