from e2b_code_interpreter import Sandbox
from langgraph.types import Command
from loguru import logger

from src.graph.models.programmer_state import DataThread, ProgrammerState
from src.modules import execute_code


def execute_code_node(state: ProgrammerState) -> dict:
    logger.info("|--> execute_code")
    thread_id = len(state["data_threads"])
    threads = state["data_threads"]
    thread = threads[-1]
    sandbox = Sandbox.connect(state["sandbox_id"])
    original_data_thread = execute_code(
        sandbox=sandbox,
        process_id="process_id",
        thread_id=thread_id,
        code=thread.code,
        user_request=thread.user_request,
    )
    threads[-1] = DataThread(
        user_request=thread.user_request,
        code=thread.code,
        error=original_data_thread.error,
        stderr=original_data_thread.stderr,
        stdout=original_data_thread.stdout,
        results=original_data_thread.results,
    )
    return Command(
        goto="generate_review",
        update={
            "data_threads": threads,
            "next_node": "generate_review",
        },
    )
