import re

from jinja2 import Template
from loguru import logger

from src.llms.apis import openai
from src.llms.models import LLMResponse


PROMPT = """プログラマにおける人格シミュレーションを行うため、与えられたペルソナ要求から、そのペルソナ定義書を作成してください。
例えばペルソナ要求が "データサイエンティスト" である場合は、以下のように記述できます。

<ペルソナ定義書.例>
あなたは、データからルールを導き出し、ビジネスの意思決定を支援する優れたデータサイエンティストです。
PythonのAI・機械学習プログラミングに適した言語でデータマイニングを行うためのプログラムを開発し、データに基づいた合理的な意思決定をサポートします。
統計学などのデータ解析手法に基づいて、pandas, scikit-learn, matplotlib などのPythonライブラリを用いて大量のデータから法則性や関連性といった意味のある情報を抽出します。
</ペルソナ定義書.例>

生成対象となるペルソナは以下の通りです。
返答は "<ペルソナ定義書>\n" の文字列で開始すること。

<ペルソナ要求>
{{ role }}
</ペルソナ要求>"""


def generate_profile(
    role: str,
    model: str = "gpt-4o-mini-2024-07-18",
) -> LLMResponse:
    prompt_template = Template(source=PROMPT)
    message = prompt_template.render(role=role)
    response = openai.generate_response(
        [{"role": "user", "content": message}],
        model=model,
    )
    response.content = re.sub(r"<.*?>", "", response.content).strip()
    return response


def main() -> None:
    role = "QAエンジニア"
    response = generate_profile(role=role)
    logger.info(response.content)


if __name__ == "__main__":
    main()
