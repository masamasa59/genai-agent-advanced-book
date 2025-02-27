from typing import List, Dict, Union, TypedDict
from pydantic import BaseModel, Field

class RolePlayList(BaseModel):
    persona_list: List[str] = Field(..., description="ロールプレイ中に使用する人格のリスト。")

class Persona(BaseModel):
    role: str = Field(..., description="ロールプレイ中に使用する役割")
    occupation: str = Field(..., description="職業")
    education_level: str = Field(..., description="学歴")
    hobbies: str = Field(..., description="興味関心")
    skills: str = Field(..., description="スキルや知識")

class Improvement(BaseModel):
    content: str = Field(..., description="改善後のフレーズ")

# ステートの定義
class AgentState(TypedDict):
    request: str
    contents: List[str]
    personas: List[str]
    questionnaire: str
    report: str
    evaluations: List[Dict[str, Union[str, int]]]
    improved_contents: Union[List[str], None]

# エージェント実行結果としてAgentStateをそのまま利用
AgentResult = AgentState
