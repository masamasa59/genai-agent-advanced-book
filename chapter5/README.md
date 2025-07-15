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

<details><summary>E2B_API_KEY の設定（クリックで展開）</summary>

コードの実行環境は、AIアプリケーション向けに作られたサンドボックス型クラウド環境である [E2B](https://e2b.dev/) を使用します。

事前に [https://e2b.dev/sign-in](https://e2b.dev/sign-in) より、e2b にアクセスし、API キーを取得します。取得後は `.env` ファイルの `E2B_API_KEY` に値を格納してください。

<img src="https://i.gyazo.com/7a54ad6d72beaa6e47fad1f9e65ab69d.png">

</details>

<details><summary>OPENAI_API_KEY の設定（クリックで展開）</summary>
### OpenAI

テキスト生成は、OpenAI API を使用します。

事前に [https://platform.openai.com/api-keys](https://platform.openai.com/api-keys) より、OpenAI Platform にアクセスし、API キーを取得します。取得後は `.env` ファイルの `OPENAI_API_KEY` に値を格納してください。

<img src="https://i.gyazo.com/bdbe5fd77930add697f134cd153411c7.png">
</details></br>

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


## 5.3 実装準備

### 5.3.1 生成されたコードの実行環境のセットアップ

初めにコードを E2B Sandbox 環境で実行します。

```bash
uv run python scripts/01_e2b_sandbox.py
```

ここでは、E2B Sandbox 環境で `print` 関数が実行しています。

```python
from e2b_code_interpreter import Sandbox

with Sandbox() as sandbox:
    execution = sandbox.run_code("print('Hello World!')")
    logger.info("\n".join(execution.logs.stdout))
```

### 5.3.2 解析対象のデータセット概要

#### テンプレートエンジン Jinja2 の使い方

[Jinja2](https://jinja.palletsprojects.com/en/stable/) を用いてプロンプトを構築します。

```bash
uv run python scripts/02_jinja_template.py
```

ここではプレースホルダー型テンプレートによって、プレースホルダーに該当する変数応じてプロンプトの記述内容が変化することが確認できます。

```python
from jinja2 import Template

source = """{% if message %}メッセージがあります: {{ message }}{% endif %}"""
template = Template(source=source)
print("1.", template.render(message="hello"))
print("2.", template.render())
```

テンプレートは `.jinja` ファイルとして記述することも可能です（参考: [`load_template`](/chapter5/src/llms/utils/load_template.py)）。

#### データ概要の確認

解析対象のデータ情報を LLM に与えるために、[pandas](https://pandas.pydata.org/) を用いてデータ情報を記述します。

```bash
uv run python scripts/03_describe_dataframe.py
```

```python
# src/prompts/describe_dataframe.jinja

'''python
>>> df.info()
{{ df_info }}

>>> df.sample(5).to_markdown()
{{ df_sample }}

>>> df.describe()
{{ df_describe }}
'''
```

### 5.3.3 LLMの呼び出し

OpenAI API による LLM 呼び出しを行い、[`LLMResponse`](/chapter5/src/llms/models/llm_response.py) というデータ型を返す関数を実行して、プロフィールを生成します。

```bash
uv run python scripts/04_generate_profile.py
```

一部抜粋ですが、事前に定義された ([`generate_response`](/chapter5/src/llms/apis/openai.py)) 関数を、以下のようにして呼び出しています。

```python
prompt_template = Template(source=PROMPT)
message = prompt_template.render(role=role)
response: LLMResponse = openai.generate_response(
    [{"role": "user", "content": message}],
    model=model,
)
```


## 5.4 プログラム生成を行うシングルエージェントワークフロー

5.4節では、(1)コード生成 (2)コード実行 (3)実行結果のレビュー、を行うシングルエージェントワークフローを構築します。

<img src="https://i.gyazo.com/0c2893618ab7513cabc5387073e4d6b6.png">

コーディング過程を管理できるよう、(1)~(3) の結果を [`DataThread`](/chapter5/src/models/data_thread.py) として管理することとします。

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
    pathes: dict = Field(default_factory=dict)
```

### 5.4.1 コード生成（計画）

タスク要求と解析対象データに対して、要求を満たすコードを生成します。なおここでは「データの概要について教えて」という要求を指定しています。

```bash
uv run python scripts/05_generate_code.py
```

- 出力形式: [`Program`](/chapter5/src/models/program.py)
- コード生成の関数: [`generate_code`](/chapter5/src/modules/generate_code.py)
- プロンプト: [`generate_code.jinja`](src/prompts/generate_code.jinja)

### 5.4.2 コード実行（行動）

#### 解析対象データのアップロード

E2B Sandbox 上でデータを扱えるようにするため、[`set_dataframe`](/chapter5/src/modules/set_dataframe.py) を用いて解析対象のデータを Sandbox 環境にアップロードしています。

```python
# （抜粋）src/modules/set_dataframe.py
sandbox.files.write(remote_data_path, file_object)
return sandbox.run_code(
    f"import pandas as pd; df = pd.read_csv('{remote_data_path}')",
    timeout=timeout,
)
```

#### 生成されたコードの実行

次にコードを E2B Sandbox 上で実行します。なお以下では `print(df.shape)` というデータサイズを確認するコードで動作を確認します。

```bash
uv run python scripts/06_execute_code.py
```

上記で呼び出している [`execute_code`](/chapter5/src/modules/execute_code.py) 関数では、`Sandbox.run_code` メソッドを呼び出しています。

```python
# （抜粋）src/modules/execute_code.py
execution = sandbox.run_code(code, timeout=timeout)
results = [
    {"type": "png", "content": r.png}
    if r.png else {"type": "raw", "content": r.text}
    for r in execution.results
]
```

### 5.4.3 実行結果のレビュー（知覚）

LLM を用いて実行された結果に対するレビューを行います。以下では、先ほどの `print(df.shape)` の実行結果に対するレビューを LLM が生成します。

```bash
uv run python scripts/07_generate_review.py
```

- 出力形式: [`Review`](/chapter5/src/models/review.py)
- レビュー生成の関数: [`generate_review`](/chapter5/src/modules/generate_review.py)
- プロンプト: [`generate_review.jinja`](src/prompts/generate_review.jinja)


### 5.4.4 コード生成・コード実行・実行結果のレビューのサイクル実行

ここまでの一連の処理を実行します。

```bash
uv run python scripts/08_programmer.py
```

## 5.5 データ分析レポートの作成

5.4節で作成したコードを拡張して、(1)データ分析における実行計画の立案、(2)および分析結果をまとめたレポートの作成、を行うデータ分析エージェントワークフローを構築します。

<img src="https://i.gyazo.com/9ba402980e175726be4dd7bf598c56db.png">

### 5.5.1 分析計画の立案（仮説構築）

LLM を用いて、ユーザーからの要求を構成的に分解します。

```bash
uv run python scripts/09_generate_plan.py
```

- 出力形式: [`Plan`](/chapter5/src/models/plan.py)
- 計画立案の関数: [`generate_plan`](/chapter5/src/modules/generate_plan.py)
- プロンプト: [`generate_plan.jinja`](/chapter5/src/prompts/generate_plan.jinja)


### 5.5.2 プログラムの実行

5.5.1 で生成された分析計画の各サブタスクをそれぞれ実行します。

```bash
uv run python scripts/10_execute_plan.py
```

### 5.5.3 実行結果を反映したレポート生成

最後に計画に対する分析結果を見やすくまとめたレポートを生成します。
分析結果は `outputs/{process_id}/report.md` に保存されます。

```bash
uv run python scripts/11_generate_report.py \
    --data_file "data/sample.csv" \
    --user_request "scoreを最大化するための広告キャンペーンを検討したい" \
    --process_id "sample"
```


## LangGraph による実行

上記について、LangGraph で実行する


### 5.4 プログラム生成を行うシングルエージェントワークフロー

```bash
uv run python src/graph/programmer.py
```
