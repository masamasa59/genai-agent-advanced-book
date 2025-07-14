from pydantic import BaseModel, Field


class DataThread(BaseModel):
    process_id: str
    thread_id: int
    user_request: str | None
    code: str | None = None
    error: str | None = None
    stderr: str | None = None
    stdout: str | None = None
    is_completed: bool = False
    observation: str | None = None
    results: list[dict] = Field(default_factory=list)
    pathes: dict = Field(default_factory=dict)
