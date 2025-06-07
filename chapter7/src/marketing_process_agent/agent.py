from typing import Dict, List, Any
from langgraph.graph import END, START, StateGraph
from langgraph.pregel import Pregel

from configs import Settings
from custom_logger import setup_logger
from models import Task, PlanList, AgentState, AgentResult, Feedback
from prompts import PLANNER_PROMPT,CRITIC_PROMPT
from langchain_openai import AzureChatOpenAI

# Import the separated functions from tools
from tools.task_tools import (
    register_task_run,
    persona_create_run,
    customer_journey_create_run,
    content_idea_create_run,
)

# JIRA関連のインポート
from jira import JIRA

logger = setup_logger(__name__)

# PlannerAgentのクラス定義
class PlannerAgent:
    def __init__(self, jira_client, client_plan):
        self.jira_client = jira_client
        self.JIRA_PROJECT_KEY = Settings().JIRA_PROJECT_KEY
        # MarketingProcessAgentから渡されたclient_planを利用
        self.client_plan = client_plan

    def get_tasks(self) -> Dict[str, Any]:
        """
        JIRAから既存タスクを取得します。
        Returns:
            dict: {"status": "success" or "error", "existing_tasks": [Task, ...]}
        """
        try:
            issues = self.jira_client.search_issues(
                f"project = {self.JIRA_PROJECT_KEY}", maxResults=False
            )
            existing_tasks_list = [
                Task(
                    task_title=issue.fields.summary,
                    description=issue.fields.description or "説明なし",
                )
                for issue in issues
            ]
            return {"status": "success", "existing_tasks": existing_tasks_list}
        except Exception as e:
            logger.error("Error while fetching tasks: %s", e)
            return {"status": "error", "existing_tasks": []}

    def run(self, state: AgentState) -> AgentState:
        """
        タスク計画を立案するエージェント
        （既存タスクを取得し、ユーザーリクエストに基づく計画を生成する）
        """
        logger.info("Plannerエージェント開始")
        user_request = state["user_request"]

        tasks_result = self.get_tasks()
        existing_tasks = tasks_result.get("existing_tasks", [])
        state["existing_tasks"] = existing_tasks

        text_messages = [
            {"role": "system", "content": PLANNER_PROMPT},
            {"role": "user", "content": f"リクエスト: {user_request}"},
        ]
        act_plans = self.client_plan.invoke(text_messages).plans
        state["plan"] = act_plans
        logger.info("生成された計画: %s", act_plans)
        return state

# ActorAgentのクラス定義
class ActorAgent:
    def __init__(self):
        # 各エージェント専用の設定があればここで初期化（例：独自のログ設定など）
        self.settings = Settings()

    def run(self, state: AgentState) -> AgentState:
        """
        各計画に基づいてタスクを実行するエージェント
        （ペルソナ作成、カスタマージャーニー作成、コンテンツ案生成など）
        """
        logger.info("Actorエージェント開始")
        plan_list: List[PlanList] = state["plan"]
        tasks_list: List[Task] = state["existing_tasks"].copy()
        for plan in plan_list:
            subtask = plan.plan
            action_type = plan.action_type
            if action_type == "persona_create_task":
                persona_profile = persona_create_run(subtask, state)
                tasks_list.append(Task(task_title="ペルソナの作成", description=persona_profile))
            elif action_type == "customer_journey_create_task":
                customer_journey = customer_journey_create_run(state)
                tasks_list.append(Task(task_title="カスタマージャーニーの作成", description=customer_journey))
            elif action_type == "contents_idea_create_task":
                ideas = content_idea_create_run(subtask, state)
                for idea in ideas:
                    tasks_list.append(Task(
                        task_title=f"コンテンツ作成：{idea.phase} - 形式：{idea.content_type}",
                        description=f"コンテンツタイトル: {idea.content_title}\nコンテンツ概要: {idea.content_overview}"
                    ))
            state["tasks"] = tasks_list
        if len(tasks_list) == 0:
            state["complete"] = True
        return state

# CriticAgentのクラス定義
class CriticAgent:
    def __init__(self, feedback_client):
        # MarketingProcessAgentから渡されたfeedback_clientを利用
        self.feedback_client = feedback_client

    def run(self, state: AgentState) -> AgentState:
        """
        タスクの評価とフィードバックを行い、必要なタスクをJIRAに登録するエージェント
        """
        logger.info("Criticエージェント開始")
        tasks: List[Task] = state["tasks"]
        existing_tasks: List[Task] = state["existing_tasks"]
        feedback: Dict[str, Feedback] = {}
        is_complete = True

        for task in tasks:
            # 新規タスクかどうかをチェック
            if not any(et.task_title == task.task_title for et in existing_tasks):
                text_messages = [
                    {"role": "system", "content": CRITIC_PROMPT.format(user_request=state["user_request"])},
                    {"role": "user", "content": f"以下のタスクを評価してください。\nタスク: {task.dict()}"},
                ]
                # 評価を実行
                response: Feedback = self.feedback_client.invoke(text_messages)
                feedback[task.task_title] = response
                if response.is_pass:
                    register_task_run(task.task_title, task.description)
                else:
                    is_complete = False

        # 状態を更新
        state["feedback"] = feedback  # 各タスクのフィードバック
        state["iteration"] += 1  # イテレーション回数を増加
        state["complete"] = is_complete  # すべてのタスクが合格なら True
        logger.info("フィードバック: %s", feedback)
        return state

def should_continue(state: AgentState):
    """
    条件付きエッジ：イテレーションが3回超えるか、全タスクが合格なら終了する
    """
    if state["iteration"] > 3 or state["complete"]:
        return END
    return "planner_agent"

# --- MarketingProcessAgent ---
class MarketingProcessAgent:
    """
    各エージェントを管理するマネージャークラス
    MarketingProcessAgentの初期化時に、各種クライアントの情報（JIRA、client_plan、feedback_client）を管理・初期化し、
    エージェントに渡すように修正
    """
    def __init__(self):
        self.settings = Settings()

        # クライアント情報の初期化（MarketingProcessAgentで一元管理）
        self.jira_client = JIRA(
            server=self.settings.JIRA_API_BASE_URL,
            basic_auth=(self.settings.JIRA_USER_EMAIL, self.settings.JIRA_API_KEY)
        )
        self.azure_deployment = self.settings.AZURE_OPENAI_DEPLOYMENT

        # client_plan と feedback_client を MarketingProcessAgent 内で初期化
        self.client_plan = AzureChatOpenAI(
            model=self.azure_deployment, temperature=0
        ).with_structured_output(PlanList)

        self.feedback_client = AzureChatOpenAI(
            model=self.azure_deployment, temperature=0
        ).with_structured_output(Feedback)

        # 各エージェントにクライアント情報を渡す
        self.planner_agent = PlannerAgent(self.jira_client, self.client_plan)
        self.actor_agent = ActorAgent()
        self.critic_agent = CriticAgent(self.feedback_client)

        self.graph = self.create_graph()

    def create_graph(self) -> Pregel:
        """
        エージェントのメイングラフを作成する
        """
        workflow = StateGraph(AgentState)
        workflow.add_node("planner_agent", self.planner_agent.run)
        workflow.add_node("actor_agent", self.actor_agent.run)
        workflow.add_node("critic_agent", self.critic_agent.run)
        workflow.add_edge(START, "planner_agent")
        workflow.add_edge("planner_agent", "actor_agent")
        workflow.add_edge("actor_agent", "critic_agent")
        workflow.add_conditional_edges("critic_agent", should_continue, ["planner_agent", END])
        return workflow.compile()

    def run_agent(self, request: str) -> AgentResult:
        """
        エージェントを実行する

        Args:
            request (str): ユーザーリクエスト

        Returns:
            AgentResult: 最終状態
        """
        initial_state: AgentState = {
            "plan": [],
            "user_request": request,
            "existing_tasks": [],
            "tasks": [],
            "feedback": None,
            "complete": False,
            "iteration": 0,
        }
        final_state = self.graph.invoke(initial_state)
        return final_state

# --- 利用例 ---
if __name__ == "__main__":
    agent = MarketingProcessAgent()
    user_request = (
        "生成AIエージェント実践講座のコンテンツマーケティングを計画してください。"
        "ターゲットは生成AIエージェントを活用して業務効率化を目指すビジネスパーソンです。"
    )
    result = agent.run_agent(user_request)
    logger.info("最終実行結果: %s", result)
