import os
from base64 import b64decode
from io import BytesIO

from loguru import logger
from PIL import Image

from src.llms.apis import openai
from src.llms.models import LLMResponse
from src.llms.utils import load_template
from src.models import DataThread


def generate_report(
    data_info: str,
    user_request: str,
    process_data_threads: list[DataThread] = [],
    model: str = "gpt-4o-mini-2024-07-18",
    output_dir: str = "outputs/sample",
    template_file: str = "src/prompts/generate_report.jinja",
) -> LLMResponse:
    os.makedirs(output_dir, exist_ok=True)
    # プロンプトの構築
    template = load_template(template_file)
    system_message = template.render(data_info=data_info)
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": f"タスク要求: {user_request}"},
    ]
    ## 実行結果の追加
    for data_thread in process_data_threads:
        user_contents = [
            {"type": "input_text", "text": f"instruction: {data_thread.user_request}"},
            {"type": "input_text", "text": f"stdout: {data_thread.stdout}"},
            {"type": "input_text", "text": f"observation: {data_thread.observation}"},
        ]
        for rix, res in enumerate(data_thread.results):
            if res["type"] == "png":
                # img = Image.open(io.BytesIO(base64.decodebytes(bytes(res["content"], "utf-8"))))
                image_data = b64decode(res["content"])
                img = Image.open(BytesIO(image_data))
                image_path = (
                    f"{data_thread.process_id}_{data_thread.thread_id}_{rix}.png"
                )
                img.save(f"{output_dir}/{image_path}")
                user_contents.extend(
                    [
                        {
                            "type": "input_text",
                            "text": f'画像パス: "{image_path}", 画像:',
                        },
                        {
                            "type": "input_image",
                            "image_url": f"data:image/png;base64,{res['content']}",
                        },
                    ],
                )
            else:
                user_contents.append(
                    {"type": "text", "text": f"実行結果: {res['content']}"},
                )
        messages.append({"role": "user", "content": user_contents})

    # レポートの生成と保存
    llm_response = openai.generate_response(
        messages,
        model=model,
    )
    with open(f"{output_dir}/report.md", "w") as fo:
        fo.write(llm_response.content)
        logger.success(f"WRITE ... {fo.name}")
    return llm_response
