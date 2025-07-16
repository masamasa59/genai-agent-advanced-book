# 現場で活用するための生成AIエージェント実践入門 - Chapter 4

このディレクトリは、書籍「現場で活用するための生成AIエージェント実践入門」（講談社サイエンティフィック社）の第4章に関連するソースコードとリソースを含んでいます。

4章記載のコードを実行するためには、以下の手順に従ってください。

## 前提条件

このプロジェクトを実行するには、以下の準備が必要です：

- Python 3.12 以上
- Docker および Docker Compose
- VSCode
- VSCodeのMulti-root Workspaces機能を使用し、ワークスペースとして開いている（やり方は[こちら](../README.md)を参照）
- OpenAIのアカウントとAPIキー

また、Python の依存関係は `pyproject.toml` に記載されています。

## 環境構築

### 1. chapter4のワークスペースを開く
chapter4 ディレクトリに仮想環境を作成します。
VSCode の ターミナルの追加で`chapter4` を選択します。

### 2. uvのインストール

依存関係の解決には`uv`を利用します。
`uv`を使ったことがない場合、以下の方法でインストールしてください。

`pip`を使う場合：
```bash
pip install uv
```

MacまたはLinuxの場合：
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 3. Python 仮想環境の作成と依存関係のインストール

依存関係のインストール
```bash
uv sync
```

インストール後に作成した仮想環境をアクティブにします。

```bash
source .venv/bin/activate
```

### 4. 環境変数のセット
`.env` ファイルを作成し、以下の内容を追加します。

OpenAI APIキーを持っていない場合は、[OpenAIの公式サイト](https://platform.openai.com/)から取得してください。

```bash

```env
# OpenAI API設定
OPENAI_API_KEY=your_openai_api_key
OPENAI_API_BASE="https://api.openai.com/v1"
OPENAI_MODEL= "gpt-4o-2024-08-06"
```

### 5. 検索インデックスの構築

makeコマンドを使用します。

```bash
#コンテナの起動
make start.engine

#インデックスの構築
make create.index
```

`create.index`実行時にElasticsearchのコンテナでエラーが発生する場合は、`docker-compose.yml`の以下の行をコメントアウトしてください。
コメントアウトした場合、Elasticsearchのデータは永続化されないため、コンテナを削除した場合に再度インデックスを構築する必要があります。

```yaml
    volumes:
      - ./.rag_data/es_data:/usr/share/elasticsearch/data
```
