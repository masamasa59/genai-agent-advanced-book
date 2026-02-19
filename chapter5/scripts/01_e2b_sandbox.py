# プログラムリスト5.3: E2B Sandbox の使い方
#
# E2B Sandbox とは
# - クラウド上に「一時的な Python 実行環境」が1つ起動する
# - with Sandbox() を抜けるとその環境は終了する
# - 同じ with ブロック内では、複数回 run_code() したときに変数が引き継がれる（同一セッション）
#
# 実行結果 (execution) の主な中身
# - execution.logs.stdout   … 標準出力の行リスト
# - execution.logs.stderr   … 標準エラーの行リスト
# - execution.results       … セル出力（図やテキスト）のリスト（.png / .text）
# - execution.error         … 例外時は traceback など
#
from dotenv import load_dotenv
from e2b_code_interpreter import Sandbox
from loguru import logger


def main() -> None:
    # .env から E2B_API_KEY を読み込む（Sandbox() が内部で参照する）
    load_dotenv()

    # ここから「1つの Sandbox」が起動し、ブロックを抜けるまで生きている
    with Sandbox() as sandbox:
        # --- 1. 単純な print ---
        execution = sandbox.run_code("print('Hello World!')")
        logger.info("【1. print の stdout】\n" + "\n".join(execution.logs.stdout))

        execution2 = sandbox.run_code("print('オッパッピー')")
        logger.info("【2. 別の print の stdout】\n" + "\n".join(execution2.logs.stdout))

        # --- 3. 同じ Sandbox 内では変数が引き継がれる ---
        sandbox.run_code("x = 10")
        exec_sum = sandbox.run_code("print(x + 1)  # 上の x が使える")
        logger.info("【3. 変数の引き継ぎ】\n" + "\n".join(exec_sum.logs.stdout))

        # --- 4. エラーが出たときは stderr に入る ---
        exec_err = sandbox.run_code("print(1/0)")
        if exec_err.logs.stderr:
            logger.warning("【4. エラー時の stderr】\n" + "\n".join(exec_err.logs.stderr))
        # エラー後も Sandbox は生きているので、続けて実行できる
        exec_after = sandbox.run_code("print('エラーのあとも実行できる')")
        logger.info("【4. 続き】\n" + "\n".join(exec_after.logs.stdout))

        # --- 5. execution の中身の確認（何が取れるか） ---
        ex = sandbox.run_code("'実行結果オブジェクトの例'")
        logger.info(
            "【5. execution の構造】"
            f" stdout行数={len(ex.logs.stdout)}, stderr行数={len(ex.logs.stderr)}, "
            f" results件数={len(ex.results)}"
        )

    # with を抜けた時点で Sandbox は終了する（以降は run_code できない）


if __name__ == "__main__":
    main()
