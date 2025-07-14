from dotenv import load_dotenv
from e2b_code_interpreter import Sandbox

from src.modules.execute_code import execute_code
from src.modules.set_dataframe import set_dataframe


load_dotenv()

with Sandbox() as sandbox:
    set_dataframe(sandbox, "data/sample.csv")
    data_thread = execute_code(
        process_id="06_execute_code",
        thread_id=0,
        sandbox=sandbox,
        code="print(df.shape)",
    )
    print(data_thread.model_dump())
