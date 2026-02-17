# ノートブックの動かし方（chapter4）

このフォルダのノートブック（`entire_graph_runner.ipynb` / `flow_steps_runner.ipynb` / `tools.ipynb`）を動かすための手順です。

## 1. カーネル（Python 環境）の選択

- ノートブックは **chapter4 の仮想環境（Python 3.12 + 必要なパッケージ）** で実行してください。
- カーネル一覧に「Python 3.12 (chapter4)」や「.venv」が出ない場合は、**インタープリターのパスを直接指定**します。

### 手順

1. ノートブック右上の **「カーネル」**（または「Select Kernel」）をクリック
2. **「別のカーネルを選択」** をクリック
3. **「Python Environments...」** をクリック
4. **「インタープリターを入力するパスを入力...」** をクリック
5. 次のパスを指定（`chapter4` フォルダの絶対パスに合わせて読み替えてください）：
   ```
   <あなたの chapter4 フォルダのパス>/.venv/bin/python
   ```
   例: `/Users/あなたのユーザー名/Desktop/本 コード/genai-agent-advanced-book/chapter4/.venv/bin/python`

## 2. 実行順序（重要）

**すべてのノートブックで、必ず「いちばん上のセル」から順に実行してください。**

- **セル0（1つ目）**: `使用中: .../chapter4` と出るパス追加のセル → **ここを最初に実行**
- そのあと、import や `Settings()`、エージェント実行のセルを順に実行

「Run All」で一括実行しても構いません。セル順が守られていれば動きます。

## 3. ノートブック一覧

| ファイル | 内容 |
|----------|------|
| `entire_graph_runner.ipynb` | ヘルプデスクエージェントを最初から最後まで一括実行 |
| `flow_steps_runner.ipynb` | エージェントの流れをステップごとに実行 |
| `tools.ipynb` | マニュアル検索・QA検索ツール単体の動作確認 |

## 4. 一覧に「.venv」が出る場合

「Python Environments」を開いたときに、**「Python 3.12.x ('.venv': venv)」** や **「chapter4」** のような項目があれば、それを選んでも同じです。
