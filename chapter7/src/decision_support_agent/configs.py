import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY")
    AZURE_OPENAI_DEPLOYMENT_NAME: str = os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME")
    AZURE_OPENAI_DEPLOYMENT_NAME_MINI: str = os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME_MINI")
    AZURE_OPENAI_ENDPOINT: str = os.environ.get("AZURE_OPENAI_ENDPOINT")
    # 必要に応じて他の設定も追加可能

def get_settings() -> Settings:
    return Settings()
