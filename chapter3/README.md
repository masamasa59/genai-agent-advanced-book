# 生成AIエージェント実践入門 第3章

本プロジェクトは「生成AIエージェント実践入門」第3章のサンプルコードを実行するための環境です。

## 環境構築

setup.shスクリプトを使用して環境を構築できます：

```bash
# 環境構築スクリプトを実行
bash setup.sh
```

または以下のコマンドを手動で実行することもできます：

```bash
# uvがインストールされていない場合はインストール
curl -LsSf https://astral.sh/uv/install.sh | sh

# uvを使った環境構築
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt

# 開発依存関係のインストール
uv pip install ipykernel black flake8 isort pytest

# Jupyterカーネルの設定
python -m ipykernel install --user --name genai_ch3 --display-name "Python 3.12 (Chapter 3)"
```

## Jupyter Notebookの実行

環境構築後、以下のコマンドでJupyter Notebookを起動できます：

```bash
jupyter notebook
```

## 必要な環境変数

環境構築スクリプト(setup.sh)を実行すると、`.env.example`ファイルが`.env`にコピーされます。このファイルを編集して、必要なAPIキーなどを設定してください：

```bash
# OpenAI API設定
OPENAI_API_KEY=your_openai_api_key

# Azure OpenAI API設定（Azureを使用する場合）
AZURE_OPENAI_API_KEY=your_azure_openai_api_key
AZURE_OPENAI_ENDPOINT=your_azure_openai_endpoint
AZURE_OPENAI_API_VERSION=your_azure_openai_api_version

# PostgreSQL接続設定（SQLDatabaseChain用）
PGUSER=your_pg_user
PGPASSWORD=your_pg_password
PGHOST=your_pg_host
PGDATABASE=your_pg_database
PGPORT=5432

# Tavily API（WEB検索用）
TAVILY_API_KEY=your_tavily_api_key
```
