import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
    AZURE_OPENAI_DEPLOYMENT_NAME = os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME")
    AZURE_OPENAI_DEPLOYMENT_NAME_MINI = os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME_MINI")
    AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT")
    OPENAI_API_VERSION = os.environ.get("OPENAI_API_VERSION")
