# 5章 データ分析者を支援する
書籍「現場で活用するための生成AIエージェント実践入門」（講談社サイエンティフィック社）で利用されるソースコード

※ 有料のサービスとして、コード実行を E2B Sandbox、テキスト生成を OpenAI API を用いて行うためお金がかかります。

## 準備

```bash
git clone https://github.com/masamasa59/genai-agent-advanced-book.git
cd chapter5

export PYTHONPATH="./:$PYTHONPATH"
```

また `.env` ファイルに環境変数を設定します。
各キーの取得方法については次のサブセクションをご参照ください。

```
E2B_API_KEY=e2b_***
OPENAI_API_KEY=sk-proj-***
```

### e2b

コードの実行環境は、AIアプリケーション向けに作られたサンドボックス型クラウド環境である [E2B](https://e2b.dev/) を使用します。

事前に [https://e2b.dev/sign-in](https://e2b.dev/sign-in) より、e2b にアクセスし、API キーを取得します。取得後は `.env` ファイルの `E2B_API_KEY` に値を格納してください。

<img src="https://i.gyazo.com/7a54ad6d72beaa6e47fad1f9e65ab69d.png">

### OpenAI

テキスト生成は、OpenAI API を使用します。

事前に [https://platform.openai.com/api-keys](https://platform.openai.com/api-keys) より、OpenAI Platform にアクセスし、API キーを取得します。取得後は `.env` ファイルの `OPENAI_API_KEY` に値を格納してください。

<img src="https://i.gyazo.com/bdbe5fd77930add697f134cd153411c7.png">

## ディレクトリ構成
```
.
├── .env
├── core
│   ├── domain
│   │   ├── models
│   │   │   ├── analysis_report.py
│   │   │   ├── data_thread.py
│   │   │   ├── program.py
│   │   │   ├── reflection.py
│   │   │   └── report_task.py
│   │   ├── prompts
│   │   │   ├── describe_dataframe.jinja
│   │   │   ├── generate_analysis_report.jinja
│   │   │   ├── generate_profile.jinja
│   │   │   ├── generate_report_plan.jinja
│   │   │   └── programmer.jinja
│   │   ├── modules
│   │   │   ├── describe_dataframe.py
│   │   │   ├── execute_code.py
│   │   │   ├── generate_plans.py
│   │   │   ├── generate_report.py
│   │   │   └── programmer.py
│   ├── llms
│   │   ├── context_manager.py
│   │   ├── models
│   │   │   └── llm_response.py
│   │   └── libs
│   │       └── openai_.py
│   └── utils
│       └── load_prompt.py
├── data
│   └── sample.csv
├── main.py
├── notebooks
│   └── entire_graph_runner.ipynb
├── pyproject.toml
└── scripts
    ├── 0_setup.sh
    ├── 531_generate_code.sh
    ├── 532_execute_code.sh
    ├── 533_reflection.sh
    ├── 541_generate_report_plan.sh
    ├── 542_execute_subtask.sh
    ├── 543_generate_analysis_report.sh
    └── run_e2e.sh
```

|ファイル名|説明|
|:---|:---|
|.env|環境変数の定義が含まれています|
|core/|Pythonのソースコードが配置されています|
|core/domain/|データ分析エージェントを構成するディレクトリです|
|core/domain/models/|Pydantic で定義されるデータクラスが含まれています|
|core/domain/prompts/|プロンプトの定義が含まれています|
|core/domain/modules/|データ分析エージェントで使用するPythonの関数群が含まれています|
|core/llms/|LLM を呼び出すソースコードが配置されています|
|core/llms/libs|LLM API を提供するライブラリの LLM 呼び出しを記述します|
|core/utils/|domain によらない共通の関数群が含まれています|
|data/|解析対象のデータが含まれています|
|main.py|データ分析エージェントを End-to-end に実行する場合のスクリプトが含まれています|
|notebooks/|各種Jupyter Notebookが含まれており、個別の処理の実行確認に使用します|
|pyproject.toml|uv による依存関係管理のファイルです|
|scripts/|コマンドライン引数等を記述したシェルスクリプトが含まれます|


## 5.3 実装準備

### 5.3.1 生成されたコードの実行環境のセットアップ

初めにコードをサンドボックス環境で実行します。

```bash
uv run python scripts/01_e2b_sandbox.py
```

[scripts/01_e2b_sandbox.py](/chapter5/scripts/01_e2b_sandbox.py) のコードを見ると、E2B Sandbox 環境で `print` 関数が実行されているのが分かります。

```python
from e2b_code_interpreter import Sandbox

with Sandbox() as sandbox:
    execution = sandbox.run_code("print('Hello World!')")
    logger.info("\n".join(execution.logs.stdout))
```

### 5.3.2 解析対象のデータセット概要

#### テンプレートエンジン Jinja2 の使い方

プロンプトの構築には [Jinja2](https://jinja.palletsprojects.com/en/stable/) を使用します。

```bash
uv run python scripts/02_jinja_template.py
```

[scripts/02_jinja_template.py](/chapter5/scripts/02_jinja_template.py) のコードを見ると、プレースホルダー型テンプレート文章が `source` 変数として宣言されていることが分かります。なおここでは `message` という変数名でプレースホルダーが定義されており、`message` に値が格納されると値が表示されます。

```python
from jinja2 import Template

source = """{% if message %}メッセージがあります: {{ message }}{% endif %}"""
template = Template(source=source)
```

プレースホルダーである `message` に値を指定することで、標準出力にテンプレート文が出力されます。

```python
print("1.", template.render(message="hello"))
print("2.", template.render())
```

Jinja2 ではテンプレート文を `.jinja` ファイルとして記述することができ、[src/llms/utils/load_template.py](/chapter5/src/llms/utils/load_template.py) では以下のようにファイルから `Template` を取得しています。

```python
def load_template(template_file: str) -> Template:
    template_path = Path(template_file)
    env = Environment(loader=FileSystemLoader(template_path.parent), autoescape=True)
    return env.get_template(template_path.name)
```

#### データ概要の確認

解析対象のデータ情報を記述するファイルを [src/prompts/describe_dataframe.jinja](/chapter5/src/prompts/describe_dataframe.jinja) に記述しています。

ここでは [pandas](https://pandas.pydata.org/) によるデータ処理の関数の実行結果をテンプレートファイルに格納します。

```bash
uv run python scripts/03_describe_dataframe.py
```

上記では [src/modules/describe_dataframe.py](/chapter5/src/modules/describe_dataframe.py) で定義される `describe_dataframe` 関数を実行します。 

```python
import pandas as pd

def describe_dataframe(
    data_file: str,
    template_file: str,
) -> str:
    df = pd.read_csv(data_file)
    buf = io.StringIO()
    df.info(buf=buf)
    df_info = buf.getvalue()
    template = load_template(template_file)
    return template.render(
        df_info=df_info,
        df_sample=df.sample(5).to_markdown(),
        df_describe=df.describe().to_markdown(),
    )
```

### 5.3.3 LLMの呼び出し

OpenAI API を含む LLM API による LLM 呼び出しを行います。
まずは出力データの型を [src/llms/models/llm_response.py](/chapter5/src/llms/models/llm_response.py) に定義します。

```python
class LLMResponse(BaseModel):
    messages: list
    content: str | BaseModel 
    model: str
    created_at: int
    input_tokens: int
    output_tokens: int
    cost: float | None = Field(default=None, init=False)
```

また今回はOpenAI APIを採用するため、[src/llms/apis/openai.py](/chapter5/src/llms/apis/openai.py) に `LLMResponse` を返す `generate_response` という関数を定義しておきます。

試しに [scripts/04_generate_profile.py](/chapter5/scripts/04_generate_profile.py) ファイルを実行して、プロフィールを生成してみます。

```bash
uv run python scripts/04_generate_profile.py

# 実行結果の例
# 2025-07-14 09:00:51.784 | DEBUG    | __main__:<module>:45 - あなたは、ソフトウェアの品質を保証するために、詳細なテスト計画を立て、実行する優れたQAエンジニアです。
# テスト自動化や手動テストを駆使し、製品の機能性やパフォーマンスを評価し、バグ捕捉や改善提案を行います。 
# SeleniumやJUnitなどのテストツールを用いて、リグレッションテストやユニットテストを実施し、コードの信頼性を高めることに注力しています。 
# また、プロジェクトチームとの密接なコミュニケーションを行い、開発者に対するフィードバックを提供し、ソフトウェアの完成度を向上させる役割も果たします。
```

一部抜粋ですが、ここでは以下のように `generate_response` を呼び出しています。

```python
from src.llms.apis import openai
from src.llms.models.llm_response import LLMResponse

# ...中略
prompt_template = Template(source=PROMPT)
message = prompt_template.render(role=role)
response = openai.generate_response(
    [{"role": "user", "content": message}],
    model=model,
)
response.content = re.sub(r"<.*?>", "", response.content).strip()
```


## 5.4 プログラム生成を行うシングルエージェントワークフロー

### 5.4.1 コード生成（計画）

### 5.4.2 コード実行（行動）

### 5.4.3 実行結果のレビュー（知覚）

### 5.4.4 コード生成・コード実行・実行結果のレビューのサイクル実行


## 5.5 データ分析レポートの作成

### 5.5.1 分析結果の立案（仮説構築）

### 5.5.2 プログラムの実行

### 5.5.3 実行結果を反映したレポート生成

