import logging

def setup_logger(name: str, level: int = logging.DEBUG) -> logging.Logger:
    """
    名前付きのロガーを作成し、コンソールにログを出力するように設定。

    Args:
        name (str): ロガー名。
        level (int): ログレベル。デフォルトはDEBUG。

    Returns:
        logging.Logger: 設定されたロガー。
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # ハンドラが設定されていない場合は新たに追加
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger

# 利用例
logger = setup_logger(__name__)
logger.debug("デバッグレベルのログが出力されます。")
