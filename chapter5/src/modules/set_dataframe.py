from e2b_code_interpreter import Sandbox
from e2b_code_interpreter.models import Execution


def set_dataframe(
    sandbox: Sandbox,
    data_file: str,
    timeout: int = 1200
) -> Execution:
    remote_path = f"/home/{data_file}"
    with open(data_file, "rb") as fi:
        sandbox.files.write(remote_path, fi)
    execution = sandbox.run_code(
        f"import pandas as pd; df = pd.read_csv('{remote_path}')",
        timeout=timeout
    )
    return execution
