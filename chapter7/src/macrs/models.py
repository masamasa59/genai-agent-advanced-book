from typing import TypedDict
from pydantic import BaseModel, Field

# ステートの定義
class AgentState(TypedDict):
    user_input: str
    conversation_history: str
    exit: bool
    selected_agent: str
    current_response: str

# ルーティング用の出力構造
class Router(BaseModel):
    selected_agent_int: int = Field(
        ...,
        description="0:QuestionAgent, 1:ChitChatAgent, 2:RecommendationAgent"
    )
