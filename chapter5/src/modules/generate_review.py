from src.llms.apis import openai
from src.llms.models import LLMResponse
from src.llms.utils import load_template
from src.models import DataThread, Review


def generate_review(
    data_info: str,
    user_request: str,
    data_thread: DataThread,
    has_results: bool = False,
    remote_save_dir: str = "outputs/process_id/id",
    model: str = "gpt-4o-mini-2024-07-18",
    template_file: str = "src/prompts/generate_review.jinja",
) -> LLMResponse:
    template = load_template(template_file)
    system_instruction = template.render(
        data_info=data_info,
        remote_save_dir=remote_save_dir,
    )
    if has_results:
        results = [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{res['content']}"},
            }
            if res["type"] == "png"
            else {"type": "text", "text": res["content"]}
            for res in data_thread.results
        ]
    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": user_request},
        {"role": "assistant", "content": data_thread.code},
        *([{"role": "system", "content": results}] if has_results else []),
        {"role": "system", "content": f"stdout: {data_thread.stdout}"},
        {"role": "system", "content": f"stderr: {data_thread.stderr}"},
        {
            "role": "user",
            "content": "実行結果に対するフィードバックを提供してください。",
        },
    ]
    return openai.generate_response(
        messages,
        model=model,
        response_format=Review,
    )
