import io

from src.graph.models.programmer_state import ProgrammerState
from src.modules import describe_dataframe, set_dataframe


def set_dataframe_node(state: ProgrammerState) -> dict:
    with open(state["data_file"], "rb") as fi:
        file_object = io.BytesIO(fi.read())
        data_info = describe_dataframe(
            file_object=file_object,
            template_file="src/prompts/describe_dataframe.jinja",
        )
        set_dataframe(sandbox=state["sandbox"], file_object=file_object)
    return {
        "data_info": data_info,
    }
