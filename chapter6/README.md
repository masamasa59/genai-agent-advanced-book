# 第6章 情報収集者を支援する：arXiv Researcher

arXiv ResearcherはarXivのAPIを利用し、自然言語での質問に基づいて、関連する論文を検索し、その内容を簡潔にまとめます。

## セットアップ

※ [uv](https://github.com/astral-sh/uv)を利用しています。uvをインストールしていない場合は、以下のコマンドでインストールしてください。

```
# On macOS and Linux.
$ curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows.
$ powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# With pip.
$ pip install uv
```

### 1. `.env.sample`を`.env`にコピーし、OpenAI API Keyなどを設定してください。

```
$ cp .env.sample .env
```

### 2. 次のコマンドで、必要なパッケージをインストールします。

```
$ uv venv
$ uv sync
```

### 3. 次のコマンドでLangGraph Studioを起動します（ローカル実行）

```
$ uv run langgraph dev --no-reload
```

### 補足: サブグラフのコマンドライン上のテスト

1. PaperAnalyzerAgent（論文分析エージェント）のテスト

```
$ uv run python -m arxiv_researcher.agent.paper_analyzer_agent fixtures/2408.14317.md
```

2．PaperSearchAgent（論文検索エージェント）のテスト

```
$ uv run python -m arxiv_researcher.agent.paper_search_agent
```

## テストで使用している論文

テストで使用している論文は以下の通りです：

```bibtex
@article{dmonte2024claim,
  title={Claim Verification in the Age of Large Language Models: A Survey},
  author={Dmonte, Alphaeus and Oruche, Roland and Zampieri, Marcos and Calyam, Prasad and Augenstein, Isabelle},
  journal={arXiv preprint arXiv:2408.14317},
  year={2024}
}
```
