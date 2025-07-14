from src.llms.apis import openai
from src.llms.models import LLMResponse
from src.llms.utils import load_template
from src.models import Plan


def generate_plan(
    data_info: str,
    user_request: str,
    model: str = "gpt-4o-mini-2024-07-18",
    template_file: str = "src/prompts/generate_plan.jinja",
) -> LLMResponse:
    template = load_template(template_file)
    system_message = template.render(
        data_info=data_info,
    )
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": f"タスク要求: {user_request}"},
    ]
    return openai.generate_response(
        messages,
        model=model,
        response_format=Plan,
    )
