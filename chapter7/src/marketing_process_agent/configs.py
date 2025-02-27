import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY")
    AZURE_OPENAI_DEPLOYMENT: str = os.environ.get("AZURE_OPENAI_DEPLOYMENT")
    JIRA_API_KEY: str = os.environ.get("JIRA_API_KEY")
    JIRA_USER_EMAIL: str = os.environ.get("JIRA_USER_EMAIL")
    JIRA_TOKEN: str = os.environ.get("JIRA_TOKEN")
    JIRA_PROJECT_KEY: str = os.environ.get("JIRA_PROJECT_KEY")
    JIRA_API_BASE_URL: str = os.environ.get("JIRA_API_BASE_URL", "https://marketing-ai-agent.atlassian.net")

def get_settings() -> Settings:
    return Settings()
