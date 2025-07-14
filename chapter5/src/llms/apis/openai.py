import os

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

from src.llms.models.llm_response import LLMResponse


load_dotenv()

# https://openai.com/api/pricing/ を参照されたい
COST = {
    "o3-mini-2025-01-31": {
        "input": 1.10 / 1_000_000,
        "output": 4.40 / 1_000_000,
    },
    "gpt-4o-2024-11-20": {
        "input": 2.50 / 1_000_000,
        "output": 1.25 / 1_000_000,
    },
    "gpt-4o-mini-2024-07-18": {
        "input": 0.150 / 1_000_000,
        "output": 0.600 / 1_000_000,
    },
}


def generate_response(
    messages: list,
    model: str = "gpt-4o-2024-11-20",
    response_format: BaseModel | None = None,
) -> LLMResponse:
    assert model in COST, f"Invalid model name: {model}"
    content_idx = 1 if model.startswith(("o1", "o3")) else 0
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    # LLM呼び出し
    if response_format is None:
        # Chat Completion
        completion = client.responses.create(model=model, input=messages)
        content = completion.output[content_idx].content[0].text
    else:
        # Structured Outputs
        completion = client.responses.parse(
            model=model,
            input=messages,
            text_format=response_format,
        )
        content = completion.output[content_idx].content[0].parsed
    # コスト計算
    input_cost = completion.usage.input_tokens * COST[model]["input"]
    output_cost = completion.usage.output_tokens * COST[model]["output"]
    return LLMResponse(
        messages=messages,
        content=content,
        model=completion.model,
        created_at=completion.created_at,
        input_tokens=completion.usage.input_tokens,
        output_tokens=completion.usage.output_tokens,
        cost=input_cost + output_cost,
    )
