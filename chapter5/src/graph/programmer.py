import argparse
import sys
from pathlib import Path

from e2b_code_interpreter import Sandbox
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command
from loguru import logger

from src.graph.models.programmer_state import ProgrammerState
from src.graph.nodes import (
    execute_code_node,
    generate_code_node,
    generate_review_node,
    set_dataframe_node,
)


def _close_programmer_node(state: ProgrammerState) -> Command:
    Sandbox.kill(state["sandbox_id"])
    return Command(
        goto=END,
        update={
            "next_node": END,
        },
    )

def build_programmer_graph(close_programmer_fn: callable) -> CompiledStateGraph:
    graph = StateGraph(ProgrammerState)
    graph.add_node("set_dataframe", set_dataframe_node)
    graph.add_node("generate_code", generate_code_node)
    graph.add_node("execute_code", execute_code_node)
    graph.add_node("generate_review", generate_review_node)
    graph.add_node("close_programmer", close_programmer_fn)
    graph.set_entry_point("set_dataframe")
    return graph.compile()


def run_programmer_workflow(
    workflow: CompiledStateGraph,
    user_request: str,
    data_file: Path,
    recursion_limit: int = 15,
) -> None:
    sandbox = Sandbox(timeout=1200)
    sandbox_id = sandbox.sandbox_id
    for state in workflow.stream(
        input={
            "user_request": user_request,
            "data_file": data_file,
            "data_threads": [],
            "sandbox_id": sandbox_id,
        },
        config={"recursion_limit": recursion_limit},
    ):
        for node_name, node_state in state.items():
            match node_name:
                case "set_dataframe":
                    data_info = node_state["data_info"]
                    print(data_info)
                case "generate_code":
                    data_thread = node_state["data_threads"][-1]
                    print(data_thread.code)
                case "execute_code":
                    data_thread = node_state["data_threads"][-1]
                    if data_thread.stdout:
                        logger.info(data_thread.stdout)
                    if data_thread.stderr:
                        logger.warning(data_thread.stderr)
                    if data_thread.results:
                        print(data_thread.results)
                case "generate_review":
                    data_thread = node_state["data_threads"][-1]
                    if data_thread.is_completed:
                        logger.success(f"observation: {data_thread.observation}")
                    else:
                        logger.warning(f"observation: {data_thread.observation}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=Path, default="data/sample.csv")
    parser.add_argument(
        "--user_request",
        type=str,
        default="scoreと曜日の関係について分析してください",
    )
    parser.add_argument("--process_id", type=str, default="programmer")
    parser.add_argument("--recursion_limit", type=int, default=15)
    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stdout, level="INFO")

    workflow = build_programmer_graph(_close_programmer_node)
    run_programmer_workflow(
        workflow=workflow,
        user_request=args.user_request,
        data_file=args.data_file,
        recursion_limit=args.recursion_limit,
    )


if __name__ == "__main__":
    main()
