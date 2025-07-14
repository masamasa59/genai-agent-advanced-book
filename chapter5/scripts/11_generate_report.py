from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from e2b_code_interpreter import Sandbox
from loguru import logger

from src.models import DataThread, Plan
from src.modules import (
    describe_dataframe,
    execute_code,
    generate_code,
    generate_plan,
    generate_report,
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
    data_info = describe_dataframe(data_file)
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


data_file = "data/sample.csv"
user_request = "scoreを最大化するための広告キャンペーンを検討したい"
output_dir = "outputs/sample_20250611"
Path(output_dir).mkdir(parents=True, exist_ok=True)

# 計画生成
data_info = describe_dataframe(data_file)
response = generate_plan(
    data_info=data_info,
    user_request=user_request,
    model="gpt-4o-mini-2024-07-18",
)
plan: Plan = response.content

# 各計画の実行
with ThreadPoolExecutor() as executor:
    futures = [
        executor.submit(
            programmer_node,
            data_file=data_file,
            user_request=task.hypothesis,
            # model="o3-mini-2025-01-31",
            model="gpt-4o-2024-11-20",
            process_id=f"sample-{idx}",
            idx=idx,
        ) for idx, task in enumerate(plan.tasks)
    ]
    _results = [future.result() for future in as_completed(futures)]

# 実行結果の保存
process_data_threads = []
for idx, data_threads in sorted(_results, key=lambda x: x[0]):
    process_data_threads.append(data_threads[-1])

response = generate_report(
    data_info=data_info,
    user_request=user_request,
    process_data_threads=process_data_threads,
    model="gpt-4o-2024-11-20",
    # model="o3-mini-2025-01-31",
    output_dir=output_dir,
)
