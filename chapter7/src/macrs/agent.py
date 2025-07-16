import asyncio
import os
import sys

from configs import Settings
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.pregel import Pregel
from models import AgentState, Router
from prompts import (
    CHITCHAT_PROMPT,
    PLANNER_PROMPT,
    QUESTION_PROMPT,
    RECOMMENDATION_PROMPT,
)

sys.path.append(
    os.path.join(os.path.dirname(__file__), "../..")
)  # srcディレクトリをパスに追加

from src.custom_logger import setup_logger

logger = setup_logger(__name__)


# 各エージェントの共通インターフェース
class BaseAgent:
    async def run(self, state: dict) -> dict:
        raise NotImplementedError("runメソッドを実装してください。")


# ユーザー入力エージェント
class UserInputAgent(BaseAgent):
    async def run(self, state: dict, prompt="あなた: ") -> dict:
        user_input = input(prompt)
        logger.info("ユーザー : %s", user_input)
        if user_input.lower() == "exit":
            print("対話を終了します。ありがとうございました！")
            state["exit"] = True
            return state
        else:
            state["conversation_history"] += f"\nユーザー: {user_input}"
            state["user_input"] = user_input
            state["exit"] = False
        return state


# 各応用エージェントクラスの定義


# 質問生成エージェント
class QuestionAgent(BaseAgent):
    def __init__(self, client: ChatOpenAI):
        self.client = client

    async def run(self, state: dict) -> dict:
        prompt = QUESTION_PROMPT
        messages = [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": "ユーザーとの過去の会話履歴："
                + state["conversation_history"],
            },
        ]
        response = await self.client.ainvoke(messages)
        question = response.content
        logger.info("AIエージェント (質問) : %s", question)
        state["conversation_history"] += "\n質問: " + question
        return state


# レコメンデーション生成エージェント
class RecommendationAgent(BaseAgent):
    def __init__(self, client: ChatOpenAI):
        self.client = client

    async def run(self, state: dict) -> dict:
        prompt = RECOMMENDATION_PROMPT
        messages = [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": "ユーザーとの過去の会話履歴："
                + state["conversation_history"],
            },
        ]
        response = await self.client.ainvoke(messages)
        recommendation = response.content
        logger.info("AIエージェント (レコメンド) : %s", recommendation)
        state["recommendation"] = recommendation
        state["conversation_history"] += "\nレコメンド: " + recommendation
        return state


# 雑談エージェント
class ChitChatAgent(BaseAgent):
    def __init__(self, client: AzureChatOpenAI):
        self.client = client

    async def run(self, state: dict) -> dict:
        messages = [
            {"role": "system", "content": CHITCHAT_PROMPT},
            {
                "role": "user",
                "content": "ユーザーとの過去の会話履歴："
                + state["conversation_history"],
            },
        ]
        response = await self.client.ainvoke(messages)
        chitchat = response.content
        logger.info("AIエージェント (雑談) : %s", chitchat)
        state["conversation_history"] += "\nおしゃべり: " + chitchat
        return state


# プランナーエージェント
class PlannerAgent(BaseAgent):
    def __init__(self, client_router: AzureChatOpenAI):
        self.client_router = client_router

    async def run(self, state: dict) -> dict:
        messages = [
            {"role": "system", "content": PLANNER_PROMPT},
            {
                "role": "user",
                "content": f"ユーザーの入力: {state['user_input']}\n会話履歴: {state['conversation_history']}",
            },
        ]
        response = await self.client_router.ainvoke(messages)
        selected_agent_int = response.selected_agent_int
        agents = ["QuestionAgent", "ChitChatAgent", "RecommendationAgent"]
        selected_agent = agents[selected_agent_int]
        state["selected_agent"] = selected_agent
        logger.info("選択されたエージェント: %s", selected_agent)
        return state


# MACRSクラス：各エージェントを管理
class MACRS:
    def __init__(self):
        load_dotenv()
        self.settings = Settings()
        self.deployment_name = self.settings.AZURE_OPENAI_DEPLOYMENT_NAME

        # Azure Chat OpenAI クライアントのセットアップ
        self.client = AzureChatOpenAI(
            azure_deployment=self.deployment_name,
            verbose=False,
            max_tokens=1024,
            temperature=0,
        )
        self.client_router = AzureChatOpenAI(
            model=self.deployment_name, temperature=0.7
        ).with_structured_output(Router)

        # 各エージェントのインスタンス化
        self.user_input_agent = UserInputAgent()
        self.question_agent = QuestionAgent(self.client)
        self.recommendation_agent = RecommendationAgent(self.client)
        self.chitchat_agent = ChitChatAgent(self.client)
        self.planner_agent = PlannerAgent(self.client_router)

    def create_graph(self) -> Pregel:
        """エージェントのメイングラフを作成する

        Returns:
            Pregel: エージェントのメイングラフ
        """
        workflow = StateGraph(AgentState)
        # 各エージェントの run メソッドをノードとして追加
        workflow.add_node("get_user_input", self.user_input_agent.run)
        workflow.add_node("QuestionAgent", self.question_agent.run)
        workflow.add_node("RecommendationAgent", self.recommendation_agent.run)
        workflow.add_node("ChitChatAgent", self.chitchat_agent.run)
        workflow.add_node("planner_agent", self.planner_agent.run)

        # エントリーポイントの設定とエッジの接続
        workflow.set_entry_point("get_user_input")

        # exitがTrueならば処理を終了、そうでなければplannerに進む
        workflow.add_conditional_edges(
            "get_user_input",
            lambda state: "exit" if state.get("exit") else "continue",
            {"exit": END, "continue": "planner_agent"},
        )

        workflow.add_conditional_edges(
            "planner_agent",
            lambda state: state["selected_agent"],
            path_map={
                "QuestionAgent": "QuestionAgent",
                "ChitChatAgent": "ChitChatAgent",
                "RecommendationAgent": "RecommendationAgent",
            },
        )
        return workflow.compile()

    async def run_agent(self):
        """エージェントの実行

        Returns:
            dict: 最終的な状態の辞書
        """

        app = self.create_graph()

        state = {
            "user_input": "",
            "conversation_history": "",
            "exit": False,
            "selected_agent": "",
        }
        print(
            "タスク管理エージェントへようこそ！操作を開始してください（終了するには 'exit' と入力してください）。"
        )
        # 初回実行
        result = await app.ainvoke(state)
        # exitフラグがセットされていれば処理を中断
        if result.get("exit"):
            return result
        state.update(result)
        # 対話のループ
        while not state.get("exit"):
            result = await app.ainvoke(state)
            if result.get("exit"):
                state.update(result)  # exitがTrueの場合も状態を更新
                break
            state.update(result)

        # 最終的な状態を返す
        return state


if __name__ == "__main__":
    asyncio.run(MACRS().run_agent())
