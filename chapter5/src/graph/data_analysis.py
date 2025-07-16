import argparse
import sys
from pathlib import Path

from e2b_code_interpreter import Sandbox
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command
from loguru import logger

from src.graph.models.data_analysis_state import DataAnalysisState
from src.graph.models.programmer_state import ProgrammerState
from src.graph.nodes import (
    approve_plan,
    generate_plan_node,
    generate_report_node,
)
from src.graph.programmer import build_programmer_graph


def open_programmer(state: DataAnalysisState) -> Command:
    logger.info("|--> open_programmer")
    sub_tasks = state["sub_tasks"]
    target_task = None
    for sub_task in sub_tasks:
        if sub_task.state is False:
            target_task = sub_task.task
            break
    # 全てのタスクが完了していたら
    if target_task is None:
        return Command(
            goto="generate_report",
            update={
                "next_node": "generate_report",
            },
        )
    # 未完了タスクがある場合は、そのタスクを実行
    user_request = target_task.purpose
    sandbox = Sandbox(timeout=1200)
    sandbox_id = sandbox.sandbox_id
    return Command(
        goto="programmer",
        update={
            "user_approval": True,
            "next_node": "programmer",
            "user_request": user_request,
            "sandbox_id": sandbox_id,
        },
    )

def _close_programmer(state: ProgrammerState) -> Command:
    logger.info("|--> _close_programmer")
    Sandbox.kill(state["sandbox_id"])
    # TODO: 最終実施タスクのデータスレッドのみを使用
    sub_task_threads = state.get("sub_task_threads", [])
    sub_task_threads.append(state["data_threads"][-1])
    return Command(
        graph=Command.PARENT,
        goto="open_programmer",
        update={
            "next_node": "open_programmer",
            "sub_task_threads": sub_task_threads,
            "data_threads": [],  # 初期化
            "sub_tasks": state.get("sub_tasks", [])[1:],  # 更新
        },
    )

def build_data_analysis_graph() -> CompiledStateGraph:
    checkpointer = InMemorySaver()
    graph = StateGraph(DataAnalysisState)
    graph.add_node("generate_plan", generate_plan_node)
    graph.add_node("approve_plan", approve_plan)
    graph.add_node("programmer", build_programmer_graph(_close_programmer))
    graph.add_node("open_programmer", open_programmer)
    graph.add_node("generate_report", generate_report_node)
    graph.set_entry_point("generate_plan")
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
        user_input = str(input("User Feedback: Approval? (y/n): "))
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
        "--user_goal",
        type=str,
        default="scoreと曜日の関係について分析してください",
    )
    parser.add_argument("--recursion_limit", type=int, default=30)
    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stdout, level="DEBUG")

    workflow = build_data_analysis_graph()
    result = invoke_workflow(
        workflow=workflow,
        input_data={
            "user_goal": args.user_goal,
            "data_file": args.data_file,
        },
        config={
            "configurable": {"thread_id": "some_id"},
            "recursion_limit": args.recursion_limit,
        },
    )

    print(result.get("report"))


if __name__ == "__main__":
    main()
