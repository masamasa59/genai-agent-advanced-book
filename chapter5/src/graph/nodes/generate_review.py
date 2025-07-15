from src.graph.models.programmer_state import ProgrammerState
from src.modules import generate_review


TEMPLATE_FILE = "src/prompts/generate_review.jinja"


def generate_review_node(state: ProgrammerState) -> dict:
    thread_id = state["current_thread_id"]
    threads = state["data_threads"]
    thread = threads[thread_id]
    response = generate_review(
        user_request=thread.task_request,
        data_info=state["data_info"],
        data_thread=thread,
    )
    thread.observation = response.content.observation
    thread.is_completed = response.content.is_completed
    threads[thread_id] = thread
    return {
        "data_threads": threads,
    }
