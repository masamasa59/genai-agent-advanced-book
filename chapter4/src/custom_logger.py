import logging
import sys


def setup_logger(name, level=logging.INFO):
    """
    名前付きのロガーを作成し、コンソールにログを出力するように設定。

    Args:
        name (str): ロガー名。
        level (int): ログレベル。デフォルトはDEBUG。

    Returns:
        logging.Logger: 設定されたロガー。
    """

    logging.basicConfig(
        level=level,  # INFO 以上を表示
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],  # stdout へ
        force=True,  # 既存ハンドラを除去
    )

    logger = logging.getLogger(name)
    logger.setLevel(level)

    return logger
