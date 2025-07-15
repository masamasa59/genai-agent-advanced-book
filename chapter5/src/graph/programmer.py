import argparse
from pathlib import Path

from e2b_code_interpreter import Sandbox
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.graph.models.programmer_state import ProgrammerState
from src.graph.nodes import (
    execute_code_node,
    generate_code_node,
    generate_review_node,
    set_dataframe_node,
)


def build_programmer_graph() -> CompiledStateGraph:
    graph = StateGraph(ProgrammerState)
    graph.add_node("set_dataframe", set_dataframe_node)
    graph.add_node("generate_code", generate_code_node)
    graph.add_node("execute_code", execute_code_node)
    graph.add_node("generate_review", generate_review_node)

    graph.add_edge("set_dataframe", "generate_code")
    graph.add_edge("generate_code", "execute_code")
    graph.add_edge("execute_code", "generate_review")
    graph.add_conditional_edges(
        "generate_review",
        lambda state: state["data_threads"][-1].is_completed,
        {
            True: END,
            False: "generate_code",
        },
    )
    graph.set_entry_point("set_dataframe")
    return graph.compile()


def run_programmer_workflow(
    workflow: CompiledStateGraph,
    user_request: str,
    data_file: Path,
    process_id: str,
    recursion_limit: int = 15,
) -> None:
    with Sandbox() as sandbox:
        for state in workflow.stream(
            input={
                "user_request": user_request,
                "data_file": data_file,
                "data_threads": [],
                "sandbox": sandbox,
                "process_id": process_id,
                "current_thread_id": -1,
            },
            config={"recursion_limit": recursion_limit},
        ):
            print(state)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--data_file", type=Path, default="data/sample.csv")
    parser.add_argument(
        "-r",
        "--user_request",
        type=str,
        default="スコアを最大化するための分析をしてください",
    )
    parser.add_argument("-p", "--process_id", type=str, default="programmer")
    parser.add_argument("--recursion_limit", type=int, default=15)
    args = parser.parse_args()

    workflow = build_programmer_graph()
    run_programmer_workflow(
        workflow=workflow,
        user_request=args.user_request,
        data_file=args.data_file,
        process_id=args.process_id,
        recursion_limit=args.recursion_limit,
    )


if __name__ == "__main__":
    main()
