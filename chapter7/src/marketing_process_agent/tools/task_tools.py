# tools/task_tools.py
import os
from typing import Any, Dict, List
from jira import JIRA
from configs import get_settings
from custom_logger import setup_logger
from models import AgentState, Persona, ContentIdea, ContentIdeaList
from prompts import (
    PERSONA_SYSTEM_PROMPT,
    CUSTOMER_JOURNEY_PROMPT,
    CONTENT_IDEA_PROMPT,
)
from langchain_openai import AzureChatOpenAI

logger = setup_logger(__name__)

# --- JIRA設定 ---
settings = get_settings()
JIRA_API_BASE_URL = settings.JIRA_API_BASE_URL
JIRA_API_KEY = settings.JIRA_API_KEY
JIRA_USER_EMAIL = settings.JIRA_USER_EMAIL
JIRA_PROJECT_KEY = settings.JIRA_PROJECT_KEY

jira_client = JIRA(server=JIRA_API_BASE_URL, basic_auth=(JIRA_USER_EMAIL, JIRA_API_KEY))

# --- AzureOpenAIクライアントの初期化 ---
deployment_name = settings.AZURE_OPENAI_DEPLOYMENT
client_persona = AzureChatOpenAI(
    model=deployment_name, temperature=1
).with_structured_output(Persona) 
client_content = AzureChatOpenAI(
    model=deployment_name, temperature=0
).with_structured_output(ContentIdeaList)
client = AzureChatOpenAI(
    azure_deployment=deployment_name,
    verbose=False,
    temperature=0,
)

def register_task_run(title: str, description: str) -> Dict[str, Any]:
    """
    JIRAにタスクを登録します。
    """
    try:
        issue = jira_client.create_issue(
            project=JIRA_PROJECT_KEY,
            summary=title,
            description=description,
            issuetype={"name": "Task"},
        )
        logger.info("JIRA登録成功: %s", title)
        return {"status": "success", "issue": issue}
    except Exception as e:
        logger.error("JIRA登録エラー: %s", e)
        return {"status": "error", "message": str(e)}

def persona_create_run(subtask: str, state: AgentState) -> str:
    """
    指定のサブタスクに基づきペルソナを作成する
    """
    user_request = state["user_request"]
    text_messages = [
        {"role": "system", "content": PERSONA_SYSTEM_PROMPT},
        {"role": "user", "content": (
            f"以下の情報をもとにペルソナを作成してください。\n"
            f"- 作成するペルソナ: {subtask}\n"
            f"- ユーザーリクエスト: {user_request}\n"
        )},
    ]
    logger.info("persona_create_run messages: %s", text_messages)
    response = client_persona.invoke(text_messages)
    persona_profile = (
        f"役割：{response.role}\n"
        f"職業：{response.occupation}\n"
        f"学歴：{response.education_level}\n"
        f"目標：{response.goals}\n"
        f"課題：{response.challenges}"
    )
    return persona_profile

def customer_journey_create_run(state: AgentState) -> str:
    """
    カスタマージャーニーを作成する
    """
    user_request = state["user_request"]
    tasks = state["tasks"]
    text_messages = [
        {"role": "system", "content": CUSTOMER_JOURNEY_PROMPT},
        {"role": "user", "content": (
            f"以下のタスク情報とユーザーリクエストを基にカスタマージャーニーを作成してください。\n"
            f"- 現在のタスク情報: {tasks}\n"
            f"- ユーザーリクエスト: {user_request}\n"
        )},
    ]
    response = client.invoke(text_messages)
    return response.content

def content_idea_create_run(subtask: str, state: AgentState) -> List[ContentIdea]:
    """
    コンテンツのテーマ案を作成する
    """
    tasks = state["tasks"]
    user_request = state["user_request"]
    text_messages = [
        {"role": "system", "content": CONTENT_IDEA_PROMPT},
        {"role": "user", "content": (
            f"以下の情報をもとにコンテンツのテーマ案を作成してください。\n"
            f"- タスク情報: {tasks}\n"
            f"- タスクの内容: {subtask}\n"
            f"- ユーザーリクエスト: {user_request}\n"
        )},
    ]
    response = client_content.invoke(text_messages)
    return response.content_idea_list
