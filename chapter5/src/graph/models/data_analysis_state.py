from typing import TypedDict

from src.graph.models.programmer_state import DataThread
from src.models import SubTask


class DataAnalysisState(TypedDict):
    data_file: str
    data_info: str
    user_goal: str
    user_request: str
    sub_tasks: list[SubTask]
    data_threads: list[DataThread]
    sub_task_threads: list[DataThread]
    report: str
    user_feedback: str
    user_approval: bool
    sandbox_id: str
    next_node: str
