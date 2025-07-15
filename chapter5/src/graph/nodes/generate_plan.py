import io

from langgraph.types import Command

from src.graph.models.data_analysis_state import DataAnalysisState
from src.modules import describe_dataframe, generate_plan


TEMPLATE_FILE = "src/prompts/generate_plan.jinja"


def generate_plan_node(state: DataAnalysisState) -> dict:
    with open(state["data_file"], "rb") as fi:
        file_object = io.BytesIO(fi.read())
        data_info = describe_dataframe(
            file_object=file_object,
            template_file=TEMPLATE_FILE,
        )
    llm_response = generate_plan(
        data_info=data_info,
        user_request=state["user_request"],
    )
    plan = llm_response.content
    return Command(
        goto="approve_plan",
        update={
            "data_info": data_info,
            "plans": plan,
            "next_node": "approve_plan",
        },
    )
