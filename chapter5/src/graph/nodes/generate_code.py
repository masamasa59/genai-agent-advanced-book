import io

from langgraph.types import Command
from loguru import logger

from src.graph.models.programmer_state import DataThread, ProgrammerState
from src.modules import describe_dataframe, generate_code


TEMPLATE_FILE = "src/prompts/generate_code.jinja"


def generate_code_node(state: ProgrammerState) -> dict:
    logger.info("|--> generate_code")
    threads = state.get("data_threads", [])
    request = state["user_request"]
    if len(threads) > 0:
        request += "\n" + threads[-1].observation
    with open(state["data_file"], "rb") as fi:
        file_object = io.BytesIO(fi.read())
    data_info = describe_dataframe(
        file_object=file_object,
        template_file=TEMPLATE_FILE,
    )
    response = generate_code(
        data_info=data_info,
        user_request=request,
    )
    thread = DataThread(
        user_request=request,
        code=response.content.code,
    )
    threads.append(thread)
    return Command(
        goto="execute_code",
        update={
            "data_threads": threads,
            "next_node": "execute_code",
        },
    )
