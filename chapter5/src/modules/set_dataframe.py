import io

from e2b_code_interpreter import Sandbox
from e2b_code_interpreter.models import Execution


def set_dataframe(
    sandbox: Sandbox,
    file_object: io.BytesIO,
    timeout: int = 1200,
    remote_data_path: str = "/home/data.csv",
) -> Execution:
    sandbox.files.write(remote_data_path, file_object)
    return sandbox.run_code(
        f"import pandas as pd; df = pd.read_csv('{remote_data_path}')",
        timeout=timeout,
    )
