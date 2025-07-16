from typing import Literal

from langgraph.types import Command, interrupt
from loguru import logger

from src.graph.models.data_analysis_state import DataAnalysisState


def approve_plan(state: DataAnalysisState) -> Command[Literal["programmer", "generate_plan"]]:
    logger.info("|--> approve_plan")
    is_approval = interrupt(
        {
            "sub_tasks": state["sub_tasks"],
        },
    )
    if is_approval.lower() == "y":
        return Command(
            goto="open_programmer",
            update={
                "user_approval": True,
                "next_node": "open_programmer",
            },
        )
    return Command(
        goto="generate_plan",
        update={
            "user_approval": False,
            "next_node": "generate_plan",
        },
    )
