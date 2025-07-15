from operator import add
from typing import Annotated, TypedDict

from src.graph.models.programmer_state import DataThread
from src.models import Plan


class DataAnalysisState(TypedDict):
    data_file: str
    data_info: str
    user_request: str
    task_request: str
    plans: Plan
    data_threads: Annotated[list[list[DataThread]], add]
    report: str
    user_feedback: str
    user_approval: bool
    sandbox_id: str
    next_node: str
