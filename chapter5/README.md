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

```tree
chapter5
├── data                              # サンプルデータ
│   └── sample.csv
├── scripts                           # src 下ファイルのサンプル実行
│   └── ...
├── src
│   ├── models
│   │   ├── data_thread.py         # コード生成・実行。レビューのデータ型
│   │   ├── program.py             # 5.4.1. コード生成
│   │   ├── review.py              # 5.4.3. 実行結果のレビュー
│   │   └── plan.py                # 5.5.1. 計画立案
│   ├── modules
│   │   ├── describe_dataframe.py  # 5.3.2. データセット概要
│   │   ├── generate_code.py       # 5.4.1. コード生成
│   │   ├── set_dataframe.py       # 5.4.2. Sandboxにデータをアップロード
│   │   ├── execute_code.py        # 5.4.2. コード実行
│   │   ├── generate_review.py     # 5.4.3. 実行結果のレビュー
│   │   ├── generate_plan.py       # 5.5.1. 計画立案
│   │   └── generate_report.py     # 5.5.3. レポート生成
│   ├── prompts
│   │   └── ...
│   └── llms
│       ├── apis
│       │   └── openai.py          # OpenAI API
│       ├── models
│       │   └── llm_response.py    # LLM APIからの出力データ型
│       └── utils
│           └── load_prompt.py     # テンプレートファイルの読み込み
├── .env                           # 環境変数の定義
├── main.py                        # データ分析エージェントの実行スクリプト
└── pyproject.toml                 # 依存関係の管理
```

### ライブラリ

[pyproject.toml](/chapter5/pyproject.toml) の `project.dependencies` は以下の通りです。

```toml
dependencies = [
    "e2b-code-interpreter>=1.1.0",
    "jinja2>=3.1.6",
    "langgraph>=0.3.11",
    "loguru>=0.7.3",
    "openai>=1.66.3",
    "pandas>=2.2.3",
    "pillow>=11.1.0",
    "pydantic>=2.10.6",
    "python-dotenv>=1.0.1",
    "tabulate>=0.9.0",
]
```

以下を実行して、依存パッケージをインストールしてください。

```bash
uv sync
```

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

本説では、コード生成・コード実行・実行結果のレビューを行う「プログラマー」としてのシングルエージェントワークフローを構築します。

<img src="https://i.gyazo.com/0c2893618ab7513cabc5387073e4d6b6.png">

プログラマーエージェントによるコーディング過程を管理できるよう、コード生成、コード実行、実行結果のレビューの一連の結果を `DataThread` として管理します（[src/models/data_thread.py](/chapter5/src/models/data_thread.py)）。

```python
class DataThread(BaseModel):
    process_id: str
    thread_id: int
    user_request: str | None
    code: str | None = None
    error: str | None = None
    stderr: str | None = None
    stdout: str | None = None
    is_completed: bool = False
    observation: str | None = None
    results: list[dict] = Field(default_factory=list)
    pathes: dict | None = Field(default=dict(), init=False)
```

### 5.4.1 コード生成（計画）

ユーザーから入力されたデータとタスク要求に対して、その要求を満たすコードを LLM によって生成します。

まずは構造化出力用のデータ型を `Program` として [src/models/program.py](/chapter5/src/models/program.py) に定義します。

```python
class Program(BaseModel):
    achievement_condition: str = Field(description="要求の達成条件")
    execution_plan: str = Field(description="実行計画")
    code: str = Field(description="生成対象となるコード")
```

コード生成を行うスクリプトの詳細は以下をご参照ください。

<details><summary>コード生成の関数（クリックで展開）</summary>

コード生成を行う関数 `generate_code` を [src/modules/generate_code.py](/chapter5/src/modules/generate_code.py) に定義します。

```python
def generate_code(
    data_info: str,
    user_request: str,
    remote_save_dir: str = "outputs/process_id/id",
    previous_thread: DataThread | None = None,
    model: str = "gpt-4o-mini-2024-07-18",
    template_file: str = "src/prompts/generate_code.jinja",
) -> LLMResponse:
    template = load_template(template_file)
    system_message = template.render(
        data_info=data_info,
        remote_save_dir=remote_save_dir,
    )
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": f"タスク要求: {user_request}"},
    ]
    # 自己修正：レビュー結果（5.4.3項参照）があれば反映する
    if previous_thread:
        # 前のスレッドのコードを追加
        messages.append({"role": "assistant", "content": previous_thread.code})
        # 前のスレッドの標準出力と標準エラーを追加
        if previous_thread.stdout and previous_thread.stderr:
            messages.extend([
                {"role": "system", "content": f"stdout: {previous_thread.stdout}"},
                {"role": "system", "content": f"stderr: {previous_thread.stderr}"},
            ])
        # 前のスレッドの観測結果を追加
        if previous_thread.observation:
            messages.append({
                "role": "user",
                "content": f"以下を参考にして、ユーザー要求を満たすコードを再生成してください: {previous_thread.observation}",
            })
    return openai.generate_response(
        messages,
        model=model,
        response_format=Program,
    )
```

</details>

<details><summary>コード生成のプロンプト（クリックで展開）</summary>
<br/>

またLLMがコード生成を行うためのプロンプトは [src/prompts/generate_code.jinja](src/prompts/generate_code.jinja) で記述しています。

```jinja
あなたは、データから重要なインサイトを引き出し、企業の戦略的意思決定を支援する優秀なデータサイエンティストです。
豊富なデータセットを分析し、さまざまなデータ解析手法を駆使することができ、とくにPython を用いた AI や機械学習に優れています。
あなたは、データ収集から前処理、探索的データ分析、そしてモデル構築までの一連のプロセスを管理します。
具体的には、pandas や NumPy を用いたデータ操作や、scikit-learn を用いた機械学習モデルの構築、matplotlib や seaborn を用いた視覚化に取り組みます。
また、必要に応じて SQL を使用し、データベースからのデータ抽出もこなします。
あなたは周囲を巻き込むことの重要性を理解しており、データ分析の結果をチームや経営層に分かりやすく伝えるスキルを持っています。
ビジネス目標に沿ったデータ戦略を立てることで、顧客のニーズを把握し、市場の変化に適応するための効果的な意思決定を促進します。

あなたの最終目的は、ユーザーのタスク要求を満たすコードを提供することです。
ユーザーからの要求は情報が不足している可能性があることを考慮し、ユーザーの意図を推測しながらタスク達成率を最大化してください。
実行環境は安全なサンドボックスで動作しているため、任意の Python コードを実行できます。

<コード生成の制約条件>
- 参照対象のデータは df という変数で与えられています。** 与えられた df 以外のデータは作成しないこと。**
- Notebook のコマンド規則を遵守し、unix コマンドを実行する際は、マジックコマンド ! を用いること。
- セルには、論理的に正しく、同一セルの中でタスク要求を満たすようなコードを生成すること。
- グラフをプロットする場合、ユーザーが後からスタイルを調整できるようにグラフのパラメータを引数として渡すこと。
- 関数を記述する際は Google Style Python Docstrings を記述すること。
- プログラムには、ユーザーが理解しやすいようにコードコメントを残すこと。
- グラフや新しいデータは "{{ remote_save_dir }}" のディレクトリ下に適切なファイル名で保存すること。
</コード生成の制約条件>

{% if data_info %}
<解析対象のデータ情報>
解析対象となるデータ情報は以下の通りです。
{{ data_info }}
</解析対象のデータ情報>
{% endif %}
```

</details>

試しに [scripts/05_generate_code.py](/chapter5/scripts/05_generate_code.py) ファイルを実行して、コードを生成してみます。なおここでは「データの概要について教えて」という要求を指定しています。

```bash
uv run python scripts/05_generate_code.py
```

### 5.4.2 コード実行（行動）

<details><summary>解析対象データのアップロード（クリックで展開）</summary>
<br/>

まずは E2B Sandbox 上でデータを解析できるようにするため、解析対象のデータを Sandbox 環境にアップロードしておきます。

ここでは [src/modules/set_dataframe.py](/chapter5/src/modules/set_dataframe.py) で定義する `set_dataframe` 関数内で、`Sandbox.files.wirte` メソッドを用いてデータをアップロードします。

```python
from e2b_code_interpreter import Sandbox
from e2b_code_interpreter.models import Execution

def set_dataframe(
    sandbox: Sandbox,
    data_file: str,
    timeout: int = 1200
) -> Execution:
    remote_path = f"/home/{data_file}"
    with open(data_file, "rb") as fi:
        sandbox.files.write(remote_path, fi)
    execution = sandbox.run_code(
        f"import pandas as pd; df = pd.read_csv('{remote_path}')",
        timeout=timeout
    )
    return execution
```

</details>

<details><summary>生成されたコードを実行する関数（クリックで展開）</summary>
<br/>

次に LLM が生成したコードを E2B Sandbox 上で実行します。

[src/modules/execute_code.py](/chapter5/src/modules/execute_code.py) の `execute_code` では、`Sandbox.run_code` メソッドを呼び出し、その実行結果を `DataThread` に格納しています。

```python
def execute_code(
    sandbox: Sandbox,
    process_id: str,
    thread_id: int,
    code: str,
    user_request: str | None = None,
    timeout: int = 1200,
) -> DataThread:
    execution = sandbox.run_code(code, timeout=timeout)
    results = [
        {"type": "png", "content": r.png}
        if r.png else {"type": "raw", "content": r.text}
        for r in execution.results
    ]
    return DataThread(
        id=execution.execution_count,
        process_id=process_id,
        thread_id=thread_id,
        user_request=user_request,
        code=code,
        error=getattr(execution.error, "traceback", None),
        stderr="".join(execution.logs.stderr).strip(),
        stdout="".join(execution.logs.stdout).strip(),
        results=results,
    )
```

</details>

[scripts/06_execute_code.py](/chapter5/scripts/06_execute_code.py) ファイルを実行して、コードを実行してみます。ここでは `print(df.shape)` というデータサイズを確認するコードを用いて動作を確認します。

```bash
uv run python scripts/06_execute_code.py
```

### 5.4.3 実行結果のレビュー（知覚）

最後に LLM を用いて実行された結果に対するレビューを行います。

構造化出力用のデータ型を `Review` として [src/models/review.py](/chapter5/src/models/review.py) に定義します。


```python
class Review(BaseModel):
    observation: str = Field(description="コードに対するフィードバックやコメント")
    is_completed: bool = Field(description="実行結果がタスク要求を満たすか")
```

実行結果に対するレビューを生成するスクリプトの詳細は以下をご参照ください。

<details><summary>実行結果に対してレビューを生成する関数（クリックで展開）</summary>

[src/modules/generate_review.py](/chapter5/src/modules/generate_review.py) の `generate_review` 関数では、直前の実行結果を保持する `DataThread` インスタンスの値を参照して、そのレビューを生成します。

```
def generate_review(
    data_info: str,
    user_request: str,
    data_thread: DataThread,
    has_results: bool = False,
    remote_save_dir: str = "outputs/process_id/id",
    model: str = "gpt-4o-mini-2024-07-18",
    template_file: str = "src/domain/modules/prompts/generate_review.jinja",
) -> LLMResponse:
    template = load_template(template_file)
    system_instruction = template.render(
        data_info=data_info,
        remote_save_dir=remote_save_dir,
    )
    if has_results:
        results = [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{res['content']}"}}
            if res["type"] == "png" else
            {"type": "text", "text": res["content"]}
            for res in data_thread.results
        ]
    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": user_request},
        {"role": "assistant", "content": data_thread.code},
        *([{"role": "system", "content": results}] if has_results else []),
        {"role": "system", "content": f"stdout: {data_thread.stdout}"},
        {"role": "system", "content": f"stderr: {data_thread.stderr}"},
        {"role": "user", "content": "コードの修正方針を示してください。"},
    ]
    return openai.generate_response(
        messages,
        model=model,
        response_format=Review,
    )
```

</details>

<details><summary>実行結果に対するレビュー生成のプロンプト（クリックで展開）</summary>

```jinja
あなたは、データから重要なインサイトを引き出し、企業の戦略的意思決定を支援する高度なデータサイエンティストです。
豊富なデータセットを分析し、様々なデータ解析手法を駆使しており、特にPythonを用いたAIや機械学習に優れています。
あなたは、データ収集から前処理、探索的データ分析、そしてモデル構築までの一連のプロセスを管理します。
具体的には、pandasやNumPyを用いたデータ操作や、scikit-learnを用いた機械学習モデルの構築、matplotlibやSeabornを用いた視覚化に取り組みます。
また、必要に応じてSQLを使用し、データベースからのデータ抽出もこなします。
あなたは巻き込むことの重要性を理解しており、データ分析の結果をチームや経営陣に分かりやすく伝えるスキルを持っています。
ビジネス目標に沿ったデータ戦略を立てることで、顧客のニーズを把握し、市場の変化に適応するための効果的な意思決定を促進します。

あなたの最終目的は、提示されたコードがユーザのタスク要求を満たしているか判定し、フィードバックを提供することです。
ユーザーからの要求は情報が不足している可能性があることを考慮し、ユーザーの意図を推測しながらタスク達成率を最大化してください。

{% if data_info %}
解析対象となるデータ情報は以下の通りです。
{{ data_info }}
{% endif %}
```

</details>
<br/>

試しに [scripts/07_generate_review.py](/chapter5/scripts/07_generate_review.py) ファイルを実行して、先ほどの `print(df.shape)` の実行結果に対するレビューを生成してみます。

```bash
uv run python scripts/07_generate_review.py
```


### 5.4.4 コード生成・コード実行・実行結果のレビューのサイクル実行

最後にここまでの一連の処理を実行します。

```bash
uv run python scripts/08_programmer.py
```

## 5.5 データ分析レポートの作成

5.4節で作成したプログラマーエージェントを拡張して、データ分析における実行計画の立案、および分析結果をまとめたレポートの作成を行うデータ分析エージェントワークフローを構築します。

<img src="https://i.gyazo.com/9ba402980e175726be4dd7bf598c56db.png">

### 5.5.1 分析結果の立案（仮説構築）

はじめに分析計画としてのデータ型を `Plan` として定義します。

[src/models/plan.py](/chapter5/src/models/plan.py)

```python
class Task(BaseModel):
    hypothesis: str = Field(description="分析レポートにおいて検証可能な仮説")
    purpose: str = Field(description="仮説の検証目的")
    description: str = Field(description="具体的な分析方針と可視化対象")
    chart_type: str = Field(description="グラフ想定（例：ヒストグラム、棒グラフなど）")

class Plan(BaseModel):
    purpose: str = Field(description="タスク要求から解釈される問い合わせ目的")
    archivement: str = Field(description="タスク要求から推測されるタスク達成条件")
    tasks: list[Task]
```

<details><summary>分析計画の立案を実施する関数（クリックで展開）</summary>

ユーザーからのタスク要求をいくつかのサブタスクに分解します。

```python
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
```

</details>

<details><summary>分析計画の立案におけるプロンプト（クリックで展開）</summary>

```jinja
あなたは、データから重要なインサイトを引き出し、企業の戦略的意思決定を支援する高度なデータサイエンティストです。
豊富なデータセットを分析し、様々なデータ解析手法を駆使しており、特にPythonを用いたAIや機械学習に優れています。
あなたは、データ収集から前処理、探索的データ分析、そしてモデル構築までの一連のプロセスを管理します。
具体的には、pandasやNumPyを用いたデータ操作や、scikit-learnを用いた機械学習モデルの構築、matplotlibやSeabornを用いた視覚化に取り組みます。
また、必要に応じてSQLを使用し、データベースからのデータ抽出もこなします。
あなたは巻き込むことの重要性を理解しており、データ分析の結果をチームや経営陣に分かりやすく伝えるスキルを持っています。
ビジネス目標に沿ったデータ戦略を立てることで、顧客のニーズを把握し、市場の変化に適応するための効果的な意思決定を促進します。

あなたの最終目的は、ユーザから与えられた「タスク要求」をもとに、仮説を立てながら分析レポートの作成計画を練ることです。
レポートは部長との60分間の戦略会合で活用され、分析方針を明確にしたり、マーケティング戦略の方向性を探るための叩き台となります。

「タスク要求」には曖昧な部分が含まれることが多いため、それを前提に柔軟なアプローチを取る必要があります。
与えられた情報から仮説を立て、その仮説に基づいてレポートを構成することで、具体的な分析方針や戦略の手がかりを示すことを目指してください。
最終的なレポートは日本語で記述し、読み手にとってわかりやすく、実用的なものとなるよう心がけてください。

{% if data_info %}
なお解析対象となるデータ情報は以下の通りです。
{{ data_info }}
{% endif %}
```

</details>

試しに [scripts/09_generate_plan.py](/chapter5/scripts/09_generate_plan.py) ファイルを実行して、プロフィールを生成してみます。

```bash
uv run python scripts/09_generate_plan.py
```

### 5.5.2 プログラムの実行

生成された分析結果のタスクをそれぞれ実行します。

```bash
uv run python scripts/10_execute_plan.py
```

### 5.5.3 実行結果を反映したレポート生成

最後に計画に対する分析結果を見やすくまとめたレポートを生成します。

```bash
uv run python scripts/11_generate_report.py
```
