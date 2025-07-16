from typing import TypedDict

from pydantic import BaseModel, Field


class RolePlayList(BaseModel):
    persona_list: list[str] = Field(
        ..., description="ロールプレイ中に使用する人格のリスト。"
    )


class Persona(BaseModel):
    role: str = Field(..., description="ロールプレイ中に使用する役割")
    occupation: str = Field(..., description="職業")
    hobbies: str = Field(..., description="興味関心")
    skills: str = Field(..., description="スキルや知識")


class Improvement(BaseModel):
    content: str = Field(..., description="改善後のフレーズ")


# ステートの定義
class AgentState(TypedDict):
    request: str
    contents: list[str]
    personas: list[str]
    questionnaire: str
    report: str
    evaluations: list[dict[str, str | int]]
    improved_contents: list[str] | None


# エージェント実行結果としてAgentStateをそのまま利用
AgentResult = AgentState
