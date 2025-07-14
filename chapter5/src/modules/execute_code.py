from e2b_code_interpreter import Sandbox
from loguru import logger

from src.models import DataThread


def execute_code(
    sandbox: Sandbox,
    process_id: str,
    thread_id: int,
    code: str,
    user_request: str | None = None,
    timeout: int = 1200,
) -> DataThread:
    execution = sandbox.run_code(code, timeout=timeout)
    logger.debug(f"{execution=}")
    results = [
        {"type": "png", "content": r.png}
        if r.png
        else {"type": "raw", "content": r.text}
        for r in execution.results
    ]
    return DataThread(
        id=execution.execution_count,
        process_id=process_id,
        thread_id=thread_id,
        user_request=user_request,
        code=code,
        error=getattr(execution.error, "traceback", None),
        stderr="".join(execution.logs.stderr).strip(),
        stdout="".join(execution.logs.stdout).strip(),
        results=results,
    )
