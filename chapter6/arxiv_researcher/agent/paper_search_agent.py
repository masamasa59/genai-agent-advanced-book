import operator
from typing import Annotated, TypedDict

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph

from arxiv_researcher.agent.paper_analyzer_agent import (
    PaperAnalyzerAgent,
    PaperAnalyzerAgentInputState,
)
from arxiv_researcher.chains.paper_processor_chain import PaperProcessor
from arxiv_researcher.models import ReadingResult
from arxiv_researcher.searcher.arxiv_searcher import ArxivSearcher
from arxiv_researcher.settings import settings


class PaperSearchAgentInputState(TypedDict):
    goal: str
    tasks: list[str]


class PaperSearchAgentProcessState(TypedDict):
    processing_reading_results: Annotated[list[ReadingResult], operator.add]


class PaperSearchAgentOutputState(TypedDict):
    reading_results: list[ReadingResult]


class PaperSearchAgentState(
    PaperSearchAgentInputState,
    PaperSearchAgentProcessState,
    PaperSearchAgentOutputState,
):
    pass


class PaperSearchAgent:
    def __init__(self, llm: ChatOpenAI, searcher: ArxivSearcher):
        self.recursion_limit = settings.langgraph.max_recursion_limit
        self.max_workers = settings.arxiv_search_agent.max_workers
        self.llm = llm
        self.searcher = searcher
        self.paper_processor = PaperProcessor(
            searcher=self.searcher, max_workers=self.max_workers
        )
        self.paper_analyzer = PaperAnalyzerAgent(llm)
        self.graph = self._create_graph()

    def __call__(self) -> CompiledStateGraph:
        return self.graph

    def _create_graph(self) -> CompiledStateGraph:
        workflow = StateGraph(
            PaperSearchAgentState,
            input=PaperSearchAgentInputState,
            output=PaperSearchAgentOutputState,
        )
        workflow.add_node("search_papers", self.paper_processor)
        workflow.add_node("analyze_paper", self._analyze_paper)
        workflow.add_node("organize_results", self._organize_results)

        workflow.set_entry_point("search_papers")
        workflow.add_edge("analyze_paper", "organize_results")
        workflow.set_finish_point("organize_results")

        return workflow.compile()

    def _analyze_paper(self, state: PaperAnalyzerAgentInputState) -> dict:
        output = self.paper_analyzer.graph.invoke(
            state,
            config={
                "recursion_limit": self.recursion_limit,
            },
        )
        reading_result = output.get("reading_result")
        return {
            "processing_reading_results": [reading_result] if reading_result else []
        }

    def _organize_results(self, state: PaperSearchAgentState) -> dict:
        processing_reading_results = state.get("processing_reading_results", [])
        reading_results = []

        # 関連性のある論文のみをフィルタリング
        for result in processing_reading_results:
            if result and result.is_related:
                reading_results.append(result)

        return {"reading_results": reading_results}


graph = PaperSearchAgent(
    settings.fast_llm,
    ArxivSearcher(settings.fast_llm),
).graph

if __name__ == "__main__":
    searcher = ArxivSearcher(settings.fast_llm)
    agent = PaperSearchAgent(settings.fast_llm, searcher)
    initial_state: PaperSearchAgentState = {
        "goal": "LLMエージェントの評価方法について調べる",
        "tasks": [
            "2023年以降に発表された論文をarXivから調査し、最新のLLMエージェント評価用データセットを収集する。"
        ],
    }

    for event in agent.graph.stream(
        input=initial_state,
        config={"recursion_limit": settings.langgraph.max_recursion_limit},
        stream_mode="updates",
        subgraphs=True,
    ):
        parent, update_state = event

        # 実行ノードの情報を取得
        node = list(update_state.keys())[0]

        # parentが空の()でない場合
        if parent:
            print(f"{parent}: {node}")
        else:
            print(node)
