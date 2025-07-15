import argparse
import sys
from pathlib import Path

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command
from loguru import logger

from src.graph.models.data_analysis_state import DataAnalysisState
from src.graph.nodes import (
    approve_plan,
    generate_plan_node,
)
from src.graph.programmer import build_programmer_graph


def build_data_analysis_graph() -> CompiledStateGraph:
    checkpointer = InMemorySaver()
    graph = StateGraph(DataAnalysisState)
    graph.add_node("generate_plan", generate_plan_node)
    graph.add_node("approve_plan", approve_plan)
    graph.add_node("programmer", build_programmer_graph())
    graph.set_entry_point("generate_plan")
    graph.set_finish_point("programmer")
    return graph.compile(checkpointer=checkpointer)


def invoke_workflow(
    workflow: CompiledStateGraph,
    input_data: dict | Command,
    config: dict,
) -> dict:
    result = workflow.invoke(
        input=input_data,
        config=config,
    )
    logger.debug(result)
    if result["next_node"] == "approve_plan":
        user_input = str(input("y/n: "))
        return invoke_workflow(
            workflow=workflow,
            input_data=Command(resume=user_input),
            config=config,
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=Path, default="data/sample.csv")
    parser.add_argument(
        "--user_request",
        type=str,
        default="scoreと曜日の関係について分析してください",
    )
    parser.add_argument("--recursion_limit", type=int, default=15)
    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stdout, level="DEBUG")

    workflow = build_data_analysis_graph()
    result = invoke_workflow(
        workflow=workflow,
        input_data={
            "user_request": args.user_request,
            "data_file": args.data_file,
        },
        config={
            "configurable": {"thread_id": "some_id"},
        },
    )


if __name__ == "__main__":
    main()
