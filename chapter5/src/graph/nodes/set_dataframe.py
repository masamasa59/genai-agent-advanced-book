import io

from e2b_code_interpreter import Sandbox
from langgraph.types import Command
from loguru import logger

from src.graph.models.programmer_state import ProgrammerState
from src.modules import describe_dataframe, set_dataframe


def set_dataframe_node(state: ProgrammerState) -> dict:
    logger.info("|--> set_dataframe")
    with open(state["data_file"], "rb") as fi:
        file_object = io.BytesIO(fi.read())
        data_info = describe_dataframe(
            file_object=file_object,
            template_file="src/prompts/describe_dataframe.jinja",
        )
        sandbox = Sandbox.connect(state["sandbox_id"])
        set_dataframe(sandbox=sandbox, file_object=file_object)
    return Command(
        goto="generate_code",
        update={
            "data_info": data_info,
            "next_node": "generate_code",
        },
    )
