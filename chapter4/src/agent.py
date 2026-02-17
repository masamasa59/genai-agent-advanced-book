import operator
from datetime import datetime
from pathlib import Path
from typing import Annotated, Literal, Sequence, TypedDict

from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.constants import Send
from langgraph.graph import END, START, StateGraph
from langgraph.pregel import Pregel
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

from src.configs import Settings
from src.custom_logger import setup_logger

try:
    from src.custom_logger import add_file_handler
except ImportError:
    # custom_logger が古い場合のフォールバック
    import logging as _logging

    def add_file_handler(logger, filepath, level=_logging.INFO):
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        fh = _logging.FileHandler(filepath, encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(_logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        logger.addHandler(fh)
        return fh

from src.models import (
    AgentResult,
    Plan,
    ReflectionResult,
    SearchOutput,
    Subtask,
    ToolResult,
)
from src.prompts import HelpDeskAgentPrompts

MAX_CHALLENGE_COUNT = 3

logger = setup_logger(__file__)

# 全ノード共通の呼び出し通し番号（ログの [呼び出し#N] 用）
_log_call_count = 0


def _next_call_id() -> int:
    global _log_call_count
    _log_call_count += 1
    return _log_call_count


def _trunc(s, max_len: int = 80) -> str:
    if s is None:
        return "None"
    t = str(s)
    return t[:max_len] + "..." if len(t) > max_len else t


def _remove_previous_agent_file_handlers() -> None:
    """このモジュールが追加したファイルハンドラを取り除く（重複出力防止）。"""
    for h in list(logger.handlers):
        if getattr(h, "_helpdesk_agent_file_handler", False):
            logger.removeHandler(h)
            try:
                h.close()
            except Exception:
                pass


def _ensure_file_logging() -> Path | None:
    """ログをファイルにも出す設定を必ず行う。成功した場合はログファイルパスを返す。"""
    try:
        # __file__ は src/agent.py のパス → parent.parent で chapter4
        base = Path(__file__).resolve().parent.parent
        if (base / "src").exists():
            log_dir = base / "logs"
        else:
            log_dir = Path.cwd() / "logs"
        log_dir = log_dir.resolve()
        log_dir.mkdir(parents=True, exist_ok=True)

        # 同一秒で複数回呼ばれても衝突しにくいように microseconds まで付ける
        log_path = log_dir / f"agent_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.log"

        _remove_previous_agent_file_handlers()
        fh = add_file_handler(logger, str(log_path))
        setattr(fh, "_helpdesk_agent_file_handler", True)
        return log_path
    except Exception as e:
        logger.error(f"ログファイル設定に失敗しました: {e}")
        return None


class AgentState(TypedDict):
    question: str
    plan: list[str]
    current_step: int
    subtask_results: Annotated[Sequence[Subtask], operator.add]
    last_answer: str


class AgentSubGraphState(TypedDict):
    question: str
    plan: list[str]
    subtask: str
    is_completed: bool
    messages: list[ChatCompletionMessageParam]
    challenge_count: int
    tool_results: Annotated[Sequence[Sequence[SearchOutput]], operator.add]
    reflection_results: Annotated[Sequence[ReflectionResult], operator.add]
    subtask_answer: str


class HelpDeskAgent:
    def __init__(
        self,
        settings: Settings,
        tools: list = [],
        prompts: HelpDeskAgentPrompts = HelpDeskAgentPrompts(),
    ) -> None:
        self.settings = settings
        self.tools = tools
        self.tool_map = {tool.name: tool for tool in tools}
        self.prompts = prompts
        self.client = OpenAI(api_key=self.settings.openai_api_key)
        self.log_path: str | None = None

        # run_agent を使わず create_graph/app.invoke するケースでもファイルに残す
        lp = _ensure_file_logging()
        if lp is not None:
            self.log_path = str(lp)
            print(f"[agent] ログファイル: {lp}")
            logger.info(f"========== HelpDeskAgent 初期化（ログファイル: {lp}）==========")

    def create_plan(self, state: AgentState) -> dict:
        """「実行計画」（サブタスクのリスト）を LLM で作成する。

        ※ ここで作るのは LangGraph のグラフ（ノード・エッジ）ではなく、
          「この質問を解くために何をすべきか」の計画（例: サブタスク 4 件のリスト）。
        グラフそのものは create_graph() で組み立てている。

        Args:
            state (AgentState): 入力の状態（question など）

        Returns:
            dict: 更新する状態の一部。{"plan": ["サブタスク1", "サブタスク2", ...]}
        """
        cid = _next_call_id()
        q = state.get("question", "")
        logger.info(
            f"[呼び出し#{cid}] create_plan: 開始 | 入力: question_len={len(q)}, question={_trunc(q, 100)}"
        )

        # tool定義を渡しシステムプロンプトを生成
        system_prompt = self.prompts.planner_system_prompt

        # ユーザーの質問を渡しユーザープロンプトを生成
        user_prompt = self.prompts.planner_user_prompt.format(
            question=state["question"],
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        logger.debug(f"Final prompt messages: {messages}")

        # OpenAIにリクエストを送信
        try:
            logger.info("Sending request to OpenAI...")
            response = self.client.beta.chat.completions.parse(
                model=self.settings.openai_model,
                messages=messages,
                response_format=Plan,
                temperature=0,
                seed=0,
            )
            logger.info("✅ Successfully received response from OpenAI.")
        except Exception as e:
            logger.error(f"Error during OpenAI request: {e}")
            raise

        # レスポンスからStructured outputを利用しPlanクラスを取得
        plan = response.choices[0].message.parsed
        out = {"plan": plan.subtasks}

        logger.info(
            f"[呼び出し#{cid}] create_plan: 完了 | 出力: plan件数={len(out['plan'])}, "
            f"subtasks={[ _trunc(s, 40) for s in out['plan'] ]}"
        )
        return out

    def select_tools(self, state: AgentSubGraphState) -> dict:
        """ツールを選択する

        Args:
            state (AgentSubGraphState): 入力の状態

        Returns:
            dict: 更新された状態
        """
        cid = _next_call_id()
        messages_pre = state.get("messages") or []
        logger.info(
            f"[呼び出し#{cid}] select_tools: 開始 | 入力: subtask={_trunc(state.get('subtask',''), 60)}, "
            f"challenge_count={state.get('challenge_count', 0)}, messages件数={len(messages_pre)}"
        )

        # OpenAI対応のtool定義に書き換える
        logger.debug("Converting tools for OpenAI format...")
        openai_tools = [convert_to_openai_tool(tool) for tool in self.tools]

        # リトライされたかどうかでプロンプトを切り替える
        if state["challenge_count"] == 0:
            logger.debug("Creating user prompt for tool selection...")
            user_prompt = self.prompts.subtask_tool_selection_user_prompt.format(
                question=state["question"],
                plan=state["plan"],
                subtask=state["subtask"],
            )

            messages = [
                {"role": "system", "content": self.prompts.subtask_system_prompt},
                {"role": "user", "content": user_prompt},
            ]

        else:
            logger.debug("Creating user prompt for tool retry...")

            # リトライされた場合は過去の対話情報にプロンプトを追加する
            messages = list(state.get("messages") or [])

            # NOTE: トークン数節約のため過去の検索結果は除く
            # roleがtoolまたはtool_callsを持つものは除く
            messages = [message for message in messages if message["role"] != "tool" or "tool_calls" not in message]

            user_retry_prompt = self.prompts.subtask_retry_answer_user_prompt
            user_message = {"role": "user", "content": user_retry_prompt}
            messages.append(user_message)

        try:
            logger.info("Sending request to OpenAI...")
            response = self.client.chat.completions.create(
                model=self.settings.openai_model,
                messages=messages,
                tools=openai_tools,  # type: ignore
                temperature=0,
                seed=0,
            )
            logger.info("✅ Successfully received response from OpenAI.")
        except Exception as e:
            logger.error(f"Error during OpenAI request: {e}")
            raise

        raw_tool_calls = getattr(response.choices[0].message, "tool_calls", None)
        tool_calls = list(raw_tool_calls) if raw_tool_calls else []
        if not tool_calls:
            logger.warning("No tool calls returned from model. Continuing without tool execution.")

        ai_message = {
            "role": "assistant",
            "tool_calls": [tool_call.model_dump() for tool_call in tool_calls],
        }

        messages.append(ai_message)
        out = {"messages": messages}
        logger.info(
            f"[呼び出し#{cid}] select_tools: 完了 | 出力: messages件数={len(out['messages'])}, "
            f"tool_calls件数={len(tool_calls)}"
        )
        return out

    def execute_tools(self, state: AgentSubGraphState) -> dict:
        """ツールを実行する

        Args:
            state (AgentSubGraphState): 入力の状態

        Raises:
            ValueError: toolがNoneの場合

        Returns:
            dict: 更新された状態
        """
        cid = _next_call_id()
        messages = state.get("messages") or []
        tool_calls = messages[-1].get("tool_calls", []) if messages else []
        logger.info(
            f"[呼び出し#{cid}] execute_tools: 開始 | 入力: messages件数={len(messages)}, "
            f"tool_calls件数={len(tool_calls)}"
        )

        # 最後のメッセージからツールの呼び出しを取得
        if not tool_calls:
            logger.warning("No tool calls to execute. Skipping tool execution.")
            out = {"messages": messages, "tool_results": [[]]}
            logger.info(f"[呼び出し#{cid}] execute_tools: 完了 | 出力: tool_results=0件（スキップ）")
            return out

        tool_results = []

        for tool_call in tool_calls:
            tool_name = tool_call["function"]["name"]
            tool_args = tool_call["function"]["arguments"]

            tool = self.tool_map[tool_name]
            tool_result: list[SearchOutput] = tool.invoke(tool_args)

            tool_results.append(
                ToolResult(
                    tool_name=tool_name,
                    args=tool_args,
                    results=tool_result,
                )
            )

            messages.append(
                {
                    "role": "tool",
                    "content": str(tool_result),
                    "tool_call_id": tool_call["id"],
                }
            )
        out = {"messages": messages, "tool_results": [tool_results]}
        names = [tr.tool_name for tr in tool_results]
        logger.info(
            f"[呼び出し#{cid}] execute_tools: 完了 | 出力: 実行ツール={names}, "
            f"tool_resultsバッチ数=1, 内訳件数={len(tool_results)}"
        )
        return out

    def create_subtask_answer(self, state: AgentSubGraphState) -> dict:
        """サブタスク回答を作成する

        Args:
            state (AgentSubGraphState): 入力の状態

        Returns:
            dict: 更新された状態
        """
        cid = _next_call_id()
        messages = state.get("messages") or []
        logger.info(
            f"[呼び出し#{cid}] create_subtask_answer: 開始 | 入力: subtask={_trunc(state.get('subtask',''), 50)}, "
            f"messages件数={len(messages)}"
        )

        try:
            logger.info("Sending request to OpenAI...")
            response = self.client.chat.completions.create(
                model=self.settings.openai_model,
                messages=messages,
                temperature=0,
                seed=0,
            )
            logger.info("✅ Successfully received response from OpenAI.")
        except Exception as e:
            logger.error(f"Error during OpenAI request: {e}")
            raise

        subtask_answer = response.choices[0].message.content

        ai_message = {"role": "assistant", "content": subtask_answer}
        messages.append(ai_message)

        out = {"messages": messages, "subtask_answer": subtask_answer}
        logger.info(
            f"[呼び出し#{cid}] create_subtask_answer: 完了 | 出力: subtask_answer={_trunc(subtask_answer, 80)}"
        )
        return out

    def reflect_subtask(self, state: AgentSubGraphState) -> dict:
        """サブタスク回答を内省する

        Args:
            state (AgentSubGraphState): 入力の状態

        Raises:
            ValueError: reflection resultがNoneの場合

        Returns:
            dict: 更新された状態
        """
        cid = _next_call_id()
        messages = state.get("messages") or []
        logger.info(
            f"[呼び出し#{cid}] reflect_subtask: 開始 | 入力: subtask={_trunc(state.get('subtask',''), 50)}, "
            f"challenge_count={state.get('challenge_count', 0)}, messages件数={len(messages)}"
        )

        user_prompt = self.prompts.subtask_reflection_user_prompt

        messages.append({"role": "user", "content": user_prompt})

        try:
            logger.info("Sending request to OpenAI...")
            response = self.client.beta.chat.completions.parse(
                model=self.settings.openai_model,
                messages=messages,
                response_format=ReflectionResult,
                temperature=0,
                seed=0,
            )
            logger.info("✅ Successfully received response from OpenAI.")
        except Exception as e:
            logger.error(f"Error during OpenAI request: {e}")
            raise

        reflection_result = response.choices[0].message.parsed
        if reflection_result is None:
            raise ValueError("Reflection result is None")

        messages.append(
            {
                "role": "assistant",
                "content": reflection_result.model_dump_json(),
            }
        )

        update_state = {
            "messages": messages,
            "reflection_results": [reflection_result],
            "challenge_count": state["challenge_count"] + 1,
            "is_completed": reflection_result.is_completed,
        }

        if update_state["challenge_count"] >= MAX_CHALLENGE_COUNT and not reflection_result.is_completed:
            update_state["subtask_answer"] = f"{state['subtask']}の回答が見つかりませんでした。"

        logger.info(
            f"[呼び出し#{cid}] reflect_subtask: 完了 | 出力: is_completed={update_state['is_completed']}, "
            f"challenge_count={update_state['challenge_count']}, "
            f"subtask_answer有無={('subtask_answer' in update_state)}"
        )
        return update_state

    def create_answer(self, state: AgentState) -> dict:
        """最終回答を作成する

        Args:
            state (AgentState): 入力の状態

        Returns:
            dict: 更新された状態
        """
        cid = _next_call_id()
        plan_list = state.get("plan") or []
        subtask_results_list = state.get("subtask_results") or []
        logger.info(
            f"[呼び出し#{cid}] create_answer: 開始 | 入力: plan件数={len(plan_list)}, "
            f"subtask_results件数={len(subtask_results_list)}"
        )
        system_prompt = self.prompts.create_last_answer_system_prompt

        # サブタスク結果のうちタスク内容と回答のみを取得（None 対策で上で取得したリストを使用）
        subtask_results = [(result.task_name, result.subtask_answer) for result in subtask_results_list]
        user_prompt = self.prompts.create_last_answer_user_prompt.format(
            question=state["question"],
            plan=plan_list,
            subtask_results=str(subtask_results),
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            logger.info("Sending request to OpenAI...")
            response = self.client.chat.completions.create(
                model=self.settings.openai_model,
                messages=messages,
                temperature=0,
                seed=0,
            )
            logger.info("✅ Successfully received response from OpenAI.")
        except Exception as e:
            logger.error(f"Error during OpenAI request: {e}")
            raise

        last_answer = response.choices[0].message.content
        out = {"last_answer": last_answer}
        logger.info(
            f"[呼び出し#{cid}] create_answer: 完了 | 出力: last_answer={_trunc(last_answer, 100)}"
        )
        return out

    def _execute_subgraph(self, state: AgentState):
        cid = _next_call_id()
        step = state.get("current_step", 0)
        plan = state.get("plan") or []
        total = len(plan)
        subtask = plan[step] if step < len(plan) else ""
        logger.info(
            f"[呼び出し#{cid}] _execute_subgraph: 開始 | 入力: current_step={step}/{total}, "
            f"subtask={_trunc(subtask, 60)}"
        )
        subgraph = self._create_subgraph()

        result = subgraph.invoke(
            {
                "question": state["question"],
                "plan": state["plan"],
                "subtask": state["plan"][state["current_step"]],
                "current_step": state["current_step"],
                "is_completed": False,
                "challenge_count": 0,
            }
        )

        subtask_result = Subtask(
            task_name=result["subtask"],
            tool_results=result["tool_results"],
            reflection_results=result["reflection_results"],
            is_completed=result["is_completed"],
            subtask_answer=result["subtask_answer"],
            challenge_count=result["challenge_count"],
        )

        out = {"subtask_results": [subtask_result]}
        logger.info(
            f"[呼び出し#{cid}] _execute_subgraph: 完了 | 出力: task_name={result['subtask']}, "
            f"is_completed={result['is_completed']}, challenge_count={result['challenge_count']}, "
            f"subtask_answer={_trunc(result.get('subtask_answer',''), 60)}"
        )
        return out

    def _should_continue_exec_subtasks(self, state: AgentState) -> list:
        plan = state.get("plan", [])
        n = len(plan)
        logger.info(f"[ルーティング] create_plan→execute_subtasks: 並列数={n}, steps=0..{n - 1}")
        return [
            Send(
                "execute_subtasks",
                {
                    "question": state["question"],
                    "plan": state["plan"],
                    "current_step": idx,
                },
            )
            for idx, _ in enumerate(plan)
        ]

    def _should_continue_exec_subtask_flow(self, state: AgentSubGraphState) -> Literal["end", "continue"]:
        decision = "end" if (state["is_completed"] or state["challenge_count"] >= MAX_CHALLENGE_COUNT) else "continue"
        logger.info(
            f"[ルーティング] reflect_subtask→{decision} | is_completed={state['is_completed']}, "
            f"challenge_count={state['challenge_count']}"
        )
        return decision

    def _create_subgraph(self) -> Pregel:
        """サブグラフを作成する

        Returns:
            Pregel: サブグラフ
        """
        workflow = StateGraph(AgentSubGraphState)

        # ツール選択ノードを追加
        workflow.add_node("select_tools", self.select_tools)

        # ツール実行ノードを追加
        workflow.add_node("execute_tools", self.execute_tools)

        # サブタスク回答作成ノードを追加
        workflow.add_node("create_subtask_answer", self.create_subtask_answer)

        # サブタスク内省ノードを追加
        workflow.add_node("reflect_subtask", self.reflect_subtask)

        # ツール選択からスタート
        workflow.add_edge(START, "select_tools")

        # ノード間のエッジを追加
        workflow.add_edge("select_tools", "execute_tools")
        workflow.add_edge("execute_tools", "create_subtask_answer")
        workflow.add_edge("create_subtask_answer", "reflect_subtask")

        # サブタスク内省ノードの結果から繰り返しのためのエッジを追加
        workflow.add_conditional_edges(
            "reflect_subtask",
            self._should_continue_exec_subtask_flow,
            {"continue": "select_tools", "end": END},
        )

        app = workflow.compile()

        return app

    def create_graph(self) -> Pregel:
        """LangGraph の「グラフ」（ノードとエッジのワークフロー）を組み立てる。

        ここで初めて「どのノードをどの順で動かすか」が定義される。
        ノードの中身（create_plan や create_answer など）は既存のメソッドを登録しているだけ。

        Returns:
            Pregel: コンパイル済みのグラフ（app.invoke で実行するやつ）
        """
        workflow = StateGraph(AgentState)

        # ノード登録: 各メソッドが「1 ノード」としてグラフに乗る
        workflow.add_node("create_plan", self.create_plan)

        # Add the execution step
        workflow.add_node("execute_subtasks", self._execute_subgraph)

        workflow.add_node("create_answer", self.create_answer)

        workflow.add_edge(START, "create_plan")

        # From plan we go to agent
        workflow.add_conditional_edges(
            "create_plan",
            self._should_continue_exec_subtasks,
        )

        # From agent, we replan
        workflow.add_edge("execute_subtasks", "create_answer")

        workflow.set_finish_point("create_answer")

        app = workflow.compile()

        return app

    def run_agent(self, question: str) -> AgentResult:
        """エージェントを実行する

        内部では LangGraph のグラフを 1 回 invoke しており、
        その中で以下の順にノード（メソッド）が呼ばれる:

        1. create_plan(state)           … 計画を1回作成
        2. _should_continue_exec_subtasks(state)
           → 計画の数だけ並列で次へ
        3. _execute_subgraph(state)    … サブタスクごとに並列で、各サブグラフで:
              select_tools → execute_tools → create_subtask_answer → reflect_subtask
              （必要なら reflect の結果で select_tools に戻ってリトライ、最大3回）
        4. create_answer(state)        … 全サブタスク結果をまとめて最終回答を1回作成

        Args:
            question (str): 入力の質問

        Returns:
            AgentResult: エージェントの実行結果
        """
        # 実行ごとに新しいログファイルへ切り替える
        log_path = _ensure_file_logging()
        if log_path is not None:
            self.log_path = str(log_path)
            print(f"[agent] ログファイル: {log_path}")
            logger.info(f"========== run_agent 開始（ログファイル: {log_path}）==========")
        else:
            logger.info("========== run_agent 開始（ログファイル: 作成失敗）==========")
        logger.info(f"入力 question: {_trunc(question, 200)}")

        # グラフを組み立て、初期状態で 1 回だけ invoke（中で上記ノードが順次・並列で実行される）
        app = self.create_graph()
        result = app.invoke(
            {
                "question": question,
                "current_step": 0,
            }
        )

        plan_list = result.get("plan") or []
        subtask_list = result.get("subtask_results") or []
        last_ans = result.get("last_answer") or ""
        logger.info(
            f"========== run_agent 完了 | 出力: plan件数={len(plan_list)}, "
            f"subtask_results件数={len(subtask_list)}, last_answer={_trunc(last_ans, 120)}"
        )
        return AgentResult(
            question=question,
            plan=Plan(subtasks=plan_list),
            subtasks=subtask_list,
            answer=last_ans,
        )
