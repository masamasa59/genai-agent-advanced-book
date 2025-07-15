from typing import Literal

from e2b_code_interpreter import Sandbox
from langgraph.types import Command, interrupt

from src.graph.models.data_analysis_state import DataAnalysisState


def approve_plan(state: DataAnalysisState) -> Command[Literal["programmer", "generate_plan"]]:
    is_approval = interrupt(
        {
            "plan": state["plans"].model_dump(),
        },
    )
    if is_approval.lower() == "y":
        plan = state["plans"]
        sandbox = Sandbox(timeout=1200)
        sandbox_id = sandbox.sandbox_id
        return Command(
            goto="programmer",
            update={
                "user_approval": True,
                "next_node": "programmer",
                "task_request": plan.tasks[0].purpose,
                "sandbox_id": sandbox_id,
            },
        )
    return Command(
        goto="generate_plan",
        update={
            "user_approval": False,
            "next_node": "generate_plan",
        },
    )
