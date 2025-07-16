from langgraph.types import Command
from loguru import logger

from src.graph.models.programmer_state import ProgrammerState
from src.models import Review
from src.modules import generate_review


TEMPLATE_FILE = "src/prompts/generate_review.jinja"


def generate_review_node(state: ProgrammerState) -> dict:
    logger.info("|--> generate_review")
    threads = state["data_threads"]
    thread = threads[-1]
    response = generate_review(
        user_request=thread.user_request,
        data_info=state["data_info"],
        data_thread=thread,
    )
    review: Review = response.content
    thread.observation = review.observation
    thread.is_completed = review.is_completed
    threads[-1] = thread
    if review.is_completed:
        return Command(
            goto="close_programmer",
            update={
                "data_threads": threads,
                "next_node": "close_programmer",
            },
        )
    return Command(
        goto="generate_code",
        update={
            "data_threads": threads,
            "next_node": "generate_code",
        },
    )
