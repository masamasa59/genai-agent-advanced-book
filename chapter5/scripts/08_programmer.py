from e2b_code_interpreter import Sandbox
from loguru import logger

from src.models.data_thread import DataThread
from src.modules.describe_dataframe import describe_dataframe
from src.modules.execute_code import execute_code
from src.modules.generate_code import generate_code
from src.modules.generate_review import generate_review
from src.modules.set_dataframe import set_dataframe


def programmer_node(
    data_file: str,
    user_request: str,
    process_id: str,
    model: str = "gpt-4o-mini-2024-07-18",
    n_trial: int = 3,
    idx: int = 0,
) -> tuple[int, list[DataThread]]:
    data_info = describe_dataframe(data_file)
    data_threads: list[DataThread] = []
    with Sandbox() as sandbox:
        set_dataframe(sandbox, data_file)
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
            logger.debug(f"{program=}")
            # 5.4.2. コード実行
            data_thread = execute_code(
                sandbox,
                process_id=process_id,
                thread_id=thread_id,
                code=program.code,
                user_request=user_request,
            )
            logger.debug(f"{data_thread.stdout=}")
            logger.debug(f"{data_thread.stderr=}")
            # 5.4.3. レビュー生成
            response = generate_review(
                user_request=user_request,
                data_info=data_info,
                data_thread=data_thread,
                model=model,
            )
            review = response.content
            logger.debug(f"{review=}")
            # data_threadを追加
            data_thread.observation = review.observation
            data_thread.is_completed = review.is_completed
            data_threads.append(data_thread)
            # 終了条件
            if data_thread.is_completed:
                logger.success(f"{user_request=} | {review.observation=}")
                logger.info(program.code)
                break
    return idx, data_threads



if __name__ == "__main__":
    data_threads = programmer_node(
        data_file="data/sample.csv",
        user_request="データ概要について教えて",
        process_id="08_programmer",
    )
    logger.info(data_threads)
