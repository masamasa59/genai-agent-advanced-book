import logging
import sys
from pathlib import Path


def setup_logger(name, level=logging.INFO, log_file=None):
    """
    名前付きのロガーを作成し、コンソール（とオプションでファイル）にログを出力する。

    Args:
        name (str): ロガー名。
        level (int): ログレベル。デフォルトは INFO。
        log_file (str | Path | None): 指定時はこのパスにもログを出力する。

    Returns:
        logging.Logger: 設定されたロガー。
    """
    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(logging.Formatter(fmt))
        handlers.append(fh)
    logging.basicConfig(level=level, format=fmt, handlers=handlers, force=True)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger


def add_file_handler(logger, filepath, level=logging.INFO):
    """
    既存のロガーにファイルハンドラを追加する。
    ノートブックなどで「この実行からファイルにも残す」ときに使う。

    Args:
        logger (logging.Logger): ロガー。
        filepath (str | Path): ログファイルのパス。
        level (int): ファイルに出すログレベル。

    Returns:
        logging.FileHandler: 追加したハンドラ（削除用に返す）。
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(filepath, encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(fh)
    return fh
