from src.graph.models.programmer_state import DataThread, ProgrammerState
from src.modules import execute_code


def execute_code_node(state: ProgrammerState) -> dict:
    thread_id = state["current_thread_id"]
    threads = state["data_threads"]
    thread = threads[thread_id]
    original_data_thread = execute_code(
        sandbox=state["sandbox"],
        process_id=state["process_id"],
        thread_id=thread_id,
        code=thread.code,
        user_request=thread.task_request,
    )
    threads[thread_id] = DataThread(
        task_request=thread.task_request,
        code=thread.code,
        error=original_data_thread.error,
        stderr=original_data_thread.stderr,
        stdout=original_data_thread.stdout,
        results=original_data_thread.results,
    )
    return {
        "data_threads": threads,
    }
