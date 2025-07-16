from langgraph.graph import END
from langgraph.types import Command
from loguru import logger

from src.graph.models.data_analysis_state import DataAnalysisState
from src.modules import generate_report


TEMPLATE_FILE = "src/prompts/generate_review.jinja"


def generate_report_node(state: DataAnalysisState) -> dict:
    logger.info("|--> generate_report")
    llm_response = generate_report(
        data_info=state["data_info"],
        user_request=state["user_request"],
        process_data_threads=state["sub_task_threads"],
        model="gpt-4o-mini-2024-07-18",
        output_dir="outputs/graph",
    )
    return Command(
        goto=END,
        update={
            "report": llm_response.content,
            "next_node": END,
        },
    )
