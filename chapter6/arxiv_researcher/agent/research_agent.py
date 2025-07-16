from typing import Annotated, Literal, TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, interrupt

from arxiv_researcher.agent.paper_search_agent import PaperSearchAgent
from arxiv_researcher.chains.goal_optimizer_chain import GoalOptimizer
from arxiv_researcher.chains.hearing_chain import Hearing, HearingChain
from arxiv_researcher.chains.query_decomposer_chain import QueryDecomposer
from arxiv_researcher.chains.reporter_chain import Reporter
from arxiv_researcher.chains.task_evaluator_chain import TaskEvaluation, TaskEvaluator
from arxiv_researcher.models.reading import ReadingResult
from arxiv_researcher.searcher.arxiv_searcher import ArxivSearcher
from arxiv_researcher.settings import settings


class ResearchAgentInputState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


class ResearchAgentPrivateState(TypedDict):
    hearing: Hearing
    goal: str
    tasks: list[str]
    reading_results: list[ReadingResult]
    evaluation: TaskEvaluation
    final_output: str
    retry_count: int


class ResearchAgentOutputState(TypedDict):
    final_output: str


class ResearchAgentState(
    ResearchAgentInputState, ResearchAgentPrivateState, ResearchAgentOutputState
):
    pass


class ResearchAgent:
    def __init__(
        self,
        llm: ChatOpenAI = settings.llm,
        fast_llm: ChatOpenAI = settings.fast_llm,
        reporter_llm: ChatAnthropic = settings.reporter_llm,
    ) -> None:
        self.recursion_limit = settings.langgraph.max_recursion_limit
        self.max_evaluation_retry_count = (
            settings.arxiv_researcher.max_evaluation_retry_count
        )
        self.user_hearing = HearingChain(llm)
        self.goal_setting = GoalOptimizer(llm)
        self.decompose_query = QueryDecomposer(llm)
        self.paper_search_agent = PaperSearchAgent(
            fast_llm, searcher=ArxivSearcher(fast_llm)
        )
        self.evaluate_task = TaskEvaluator(llm)
        self.generate_report = Reporter(reporter_llm)
        self.graph = self._create_graph()

    def _create_graph(self) -> CompiledStateGraph:
        workflow = StateGraph(
            state_schema=ResearchAgentState,
            input=ResearchAgentInputState,
            output=ResearchAgentOutputState,
        )
        workflow.add_node("user_hearing", self.user_hearing)
        workflow.add_node("human_feedback", self._human_feedback)
        workflow.add_node("goal_setting", self.goal_setting)
        workflow.add_node("decompose_query", self.decompose_query)
        workflow.add_node("paper_search_agent", self._paper_search_agent)
        workflow.add_node("evaluate_task", self.evaluate_task)
        workflow.add_node("generate_report", self.generate_report)

        workflow.set_entry_point("user_hearing")
        workflow.set_finish_point("generate_report")

        return workflow.compile(checkpointer=MemorySaver())

    def _human_feedback(
        self, state: ResearchAgentState
    ) -> Command[Literal["user_hearing"]]:
        # 最後のメッセージを取得
        last_message = state["messages"][-1]
        # ユーザーへの質問を表示
        human_feedback = interrupt(last_message.content)
        if human_feedback is None:
            human_feedback = "そのままの条件で検索し、調査してください。"
        return Command(
            goto="user_hearing",
            update={"messages": [{"role": "human", "content": human_feedback}]},
        )

    def _paper_search_agent(
        self, state: ResearchAgentState
    ) -> Command[Literal["evaluate_task"]]:
        output = self.paper_search_agent.graph.invoke(
            input=state,
            config={"recursion_limit": settings.langgraph.max_recursion_limit},
        )
        return Command(
            goto="evaluate_task",
            update={"reading_results": output.get("reading_results", [])},
        )


graph = ResearchAgent().graph

if __name__ == "__main__":
    png_file = graph.get_graph().draw_mermaid_png()
    with open("graph.png", "wb") as f:
        f.write(png_file)
