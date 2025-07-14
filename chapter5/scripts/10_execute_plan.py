import base64
import io
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path

from loguru import logger
from PIL import Image

from scripts.programmer import programmer_node
from src.models import Plan
from src.modules import (
    describe_dataframe,
    generate_plan,
)


def main() -> None:
    data_file = "data/sample.csv"
    template_file = "src/prompts/generate_plan.jinja"
    user_request = "scoreを最大化するための広告キャンペーンを検討したい"
    output_dir = "outputs/tmp"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    # 計画生成
    with open(data_file, "rb") as fi:
        file_object = io.BytesIO(fi.read())
    data_info = describe_dataframe(file_object=file_object, template_file=template_file)
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
            )
            for idx, task in enumerate(plan.tasks)
        ]
        _results = [future.result() for future in as_completed(futures)]

    # 実行結果の保存
    for _, data_threads in sorted(_results, key=lambda x: x[0]):
        data_thread = data_threads[-1]
        output_file = f"{output_dir}/{data_thread.process_id}_{data_thread.thread_id}."
        if data_thread.is_completed:
            for i, res in enumerate(data_thread.results):
                if res["type"] == "png":
                    image = Image.open(BytesIO(base64.b64decode(res["content"])))
                    image.save(f"{output_file}_{i}.png")
                else:
                    with open(f"{output_file}_{i}.txt", "w") as f:
                        f.write(res["content"])
        else:
            logger.warning(f"{data_thread.user_request=} is not completed.")


if __name__ == "__main__":
    main()
