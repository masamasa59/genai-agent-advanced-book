import os

from dotenv import load_dotenv

load_dotenv()


class Settings:
    OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY")
    OPENAI_MODEL: str = os.environ.get("OPENAI_MODEL", "gpt-4o")
    # 必要に応じて他の設定も追加可能
