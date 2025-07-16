import os
import sys

from configs import Settings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import END, START, StateGraph
from langgraph.pregel import Pregel
from models import AgentResult, AgentState, Improvement, Persona, RolePlayList
from prompts import (
    CONTENT_IMPROVER_PROMPT,
    CONTENTS_ANALYZER_PROMPT,
    CONTENTS_EVALUATOR_PROMPT,
    CONTENTS_LIST,
    PERSONA_CREATE_SYSTEM_PROMPT,
    PERSONA_GENERATOR_PROMPT,
    QUESTIONNAIRE,
)

sys.path.append(
    os.path.join(os.path.dirname(__file__), "../..")
)  # srcディレクトリをパスに追加

from src.custom_logger import setup_logger

logger = setup_logger(__name__)


# 各エージェントのベースクラス（オプション）
class BaseAgent:
    def run(self, state: AgentState) -> AgentState:
        raise NotImplementedError("各エージェントで run() を実装してください。")


# ペルソナ生成エージェントのクラス定義
class PersonaGeneratorAgent(BaseAgent):
    def __init__(self, client_persona_role: ChatOpenAI, client_persona: ChatOpenAI):
        self.client_persona_role = client_persona_role
        self.client_persona = client_persona

    def persona_create_run(
        self, i, persona_list, persona_profile_list, user_request, target_text
    ):
        for _ in range(5):
            text_messages = [
                {"role": "system", "content": PERSONA_CREATE_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"# 作成するペルソナ：{persona_list[i]}\n\n"
                        f"以下はコンテンツ評価に必要な情報です。これらの情報をもとにペルソナを作成してください。\n"
                        f"・評価リクエスト：{user_request}\n"
                        f"・評価対象のコンテンツ：{target_text}"
                    ),
                },
            ]
            logger.info("persona_create_run messages: %s", text_messages)
            response = self.client_persona.invoke(text_messages)
            persona_profile = (
                f"役割：{response.role}\n"
                f"職業：{response.occupation}\n"
                f"趣味・関心：{response.hobbies}\n"
                f"スキルや知識：{response.skills}"
            )
            persona_profile_list.append(persona_profile)
        return persona_profile_list

    def run(self, state: AgentState) -> AgentState:
        persona_profile_list = []
        message = [
            {
                "role": "system",
                "content": PERSONA_GENERATOR_PROMPT,
            },
            {
                "role": "user",
                "content": (
                    f"ユーザーリクエスト：{state['request']}\n"
                    f"評価対象コンテンツ：{state['contents']}"
                ),
            },
        ]
        persona_list = self.client_persona_role.invoke(message).persona_list
        logger.info("persona_list: %s", persona_list)
        for i in range(len(persona_list)):
            persona_profile_list = self.persona_create_run(
                i=i,
                persona_list=persona_list,
                persona_profile_list=persona_profile_list,
                user_request=state["request"],
                target_text=state["contents"],
            )
            logger.info("生成ペルソナプロファイル: %s", persona_profile_list[i])
        state["personas"] = persona_profile_list
        return state


# コンテンツ評価エージェントのクラス定義
class ContentsEvaluatorAgent(BaseAgent):
    def __init__(self, client: ChatOpenAI):
        self.client = client

    def run(self, state: AgentState) -> AgentState:
        evaluations = []
        for persona in state["personas"]:
            logger.info("評価実施ペルソナ: %s", persona)
            message = [
                {
                    "role": "system",
                    "content": CONTENTS_EVALUATOR_PROMPT.format(
                        persona=persona, questionnaire=state["questionnaire"]
                    ),
                },
                {
                    "role": "user",
                    "content": f"以下のコンテンツを評価してください: {state['contents']}",
                },
            ]
            logger.info("contents_evaluator_agent message: %s", message)
            text_response = self.client.invoke(message)
            evaluations.append(
                {
                    "persona": persona,
                    "feedback": text_response.content,
                }
            )
        state["evaluations"] = evaluations
        return state


# コンテンツ分析エージェントのクラス定義
class ContentsAnalyzerAgent(BaseAgent):
    def __init__(self, client: ChatOpenAI):
        self.client = client

    def run(self, state: AgentState) -> AgentState:
        message = [
            {
                "role": "system",
                "content": CONTENTS_ANALYZER_PROMPT.format(personas=state["personas"]),
            },
            {
                "role": "user",
                "content": (
                    f"評価結果: {state['evaluations']}\n"
                    "上記評価結果に基づき、具体的な改善レポートを作成してください。"
                ),
            },
        ]
        text_response = self.client.invoke(message)
        state["report"] = text_response.content
        logger.info("作成レポート: %s", state["report"])
        return state


# コンテンツ改善エージェントのクラス定義
class ContentImproverAgent(BaseAgent):
    def __init__(self, client_improver: ChatOpenAI):
        self.client_improver = client_improver

    def run(self, state: AgentState) -> AgentState:
        message = [
            {
                "role": "system",
                "content": CONTENT_IMPROVER_PROMPT,
            },
            {
                "role": "user",
                "content": (
                    f"コンテンツ: {state['contents']}\n"
                    f"リクエスト: {state['request']}\n"
                    f"評価レポート: {state['report']}\n"
                    "上記に基づき、改善後のコンテンツを生成してください。"
                ),
            },
        ]
        text_response = self.client_improver.invoke(message)
        state["improved_contents"] = text_response.content
        logger.info("改善後コンテンツ: %s", state["improved_contents"])
        return state


# DecisionSupportAgent が各エージェントを管理
class DecisionSupportAgent:
    def __init__(self):
        settings = Settings()
        self.model_name = settings.OPENAI_MODEL

        # 各クライアントの初期化
        self.client = ChatOpenAI(
            model=self.model_name,
            verbose=False,
            temperature=0,
        )
        self.client_persona_role = ChatOpenAI(
            model=self.model_name, temperature=0
        ).with_structured_output(RolePlayList)
        self.client_persona = ChatOpenAI(
            model=self.model_name, temperature=1
        ).with_structured_output(Persona)
        self.client_improver = ChatOpenAI(
            model=self.model_name, temperature=0
        ).with_structured_output(Improvement)
        self.embeddings = OpenAIEmbeddings()
        self.default_questionnaire = QUESTIONNAIRE

        # 各エージェントのインスタンス化
        self.persona_generator = PersonaGeneratorAgent(
            client_persona_role=self.client_persona_role,
            client_persona=self.client_persona,
        )
        self.contents_evaluator = ContentsEvaluatorAgent(client=self.client)
        self.contents_analyzer = ContentsAnalyzerAgent(client=self.client)
        self.content_improver = ContentImproverAgent(
            client_improver=self.client_improver
        )

    def create_graph(self) -> Pregel:
        """エージェントのメイングラフを作成する

        Returns:
            Pregel: エージェントのメイングラフ
        """
        workflow = StateGraph(AgentState)
        workflow.add_node("persona_generator", self.persona_generator.run)
        workflow.add_node("contents_evaluator", self.contents_evaluator.run)
        workflow.add_node("contents_analyzer", self.contents_analyzer.run)
        workflow.add_node("content_improver", self.content_improver.run)
        workflow.add_edge(START, "persona_generator")
        workflow.add_edge("persona_generator", "contents_evaluator")
        workflow.add_edge("contents_evaluator", "contents_analyzer")
        workflow.add_edge("contents_analyzer", "content_improver")
        workflow.add_edge("content_improver", END)
        return workflow.compile()

    def run_agent(self, request: str) -> AgentResult:
        """
        エージェントを実行する
        """
        app = self.create_graph()

        initial_state: AgentState = {
            "request": request,
            "contents": CONTENTS_LIST,
            "personas": [],
            "questionnaire": self.default_questionnaire,
            "report": "",
            "evaluations": [],
            "improved_contents": None,
        }
        final_state = app.invoke(initial_state)
        return final_state


# --- 利用例 ---
if __name__ == "__main__":
    agent = DecisionSupportAgent()
    user_request = "生成AIエージェントを活用して業務効率化を目指すビジネスパーソンに興味をもってもらえるようにコンテンツのテーマを改善して"
    result = agent.run_agent(user_request)
    logger.info("最終実行結果: %s", result)
