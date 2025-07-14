import io

from e2b_code_interpreter import Sandbox
from loguru import logger

from src.models import DataThread
from src.modules import (
    describe_dataframe,
    execute_code,
    generate_code,
    generate_review,
    set_dataframe,
)


def programmer_node(
    data_file: str,
    user_request: str,
    process_id: str,
    model: str = "gpt-4o-mini-2024-07-18",
    n_trial: int = 3,
    idx: int = 0,
) -> tuple[int, list[DataThread]]:
    template_file = "src/prompts/describe_dataframe.jinja"
    with open(data_file, "rb") as fi:
        file_object = io.BytesIO(fi.read())
    data_info = describe_dataframe(file_object=file_object, template_file=template_file)
    data_threads: list[DataThread] = []
    with Sandbox() as sandbox:
        with open(data_file, "rb") as fi:
            set_dataframe(sandbox=sandbox, file_object=fi)
        for thread_id in range(n_trial):
            # 5.4.1. コード生成
            previous_thread = data_threads[-1] if data_threads else None
            response = generate_code(
                data_info=data_info,
                user_request=user_request,
                previous_thread=previous_thread,
                model=model,
            )
            program = response.content
            logger.info(program.model_dump_json())
            # 5.4.2. コード実行
            data_thread = execute_code(
                sandbox,
                process_id=process_id,
                thread_id=thread_id,
                code=program.code,
                user_request=user_request,
            )
            if data_thread.stdout:
                logger.debug(f"{data_thread.stdout=}")
            if data_thread.stderr:
                logger.warning(f"{data_thread.stderr=}")
            # 5.4.3. レビュー生成
            response = generate_review(
                user_request=user_request,
                data_info=data_info,
                data_thread=data_thread,
                model=model,
            )
            review = response.content
            logger.info(review.model_dump_json())
            # data_threadを追加
            data_thread.observation = review.observation
            data_thread.is_completed = review.is_completed
            data_threads.append(data_thread)
            # 終了条件
            if data_thread.is_completed:
                logger.success(f"{user_request=}")
                logger.success(f"{program.code=}")
                logger.success(f"{review.observation=}")
                break
    return idx, data_threads
