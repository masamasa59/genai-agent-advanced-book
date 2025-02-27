from typing import List, Dict, Union, Optional, TypedDict
from pydantic import BaseModel, Field

# タスク情報を表すクラス
class Task(BaseModel):
    task_title: str = Field(..., description="タスクのタイトル")
    description: str = Field(..., description="タスクの詳細説明")

# 計画内容を表すクラス
class Plan(BaseModel):
    plan: str = Field(..., description="計画")
    action_type: str = Field(...,description="アクションタイプ（persona_create_task,customer_journey_create_task,contents_idea_create_task）",
    )

class PlanList(BaseModel):
    plans: list[Plan] = Field(..., description="計画のリスト")

# タスクの評価やフィードバックを表すクラス
class Feedback(BaseModel):
    is_pass: bool = Field(
        ..., description="タスクが十分な場合はtrue,不十分な場合はfalse"
    )
    content: str = Field(..., description="改善点")

# ペルソナ情報
class Persona(BaseModel):
    role: str = Field(..., description="ロール")
    occupation: str = Field(..., description="職業")
    education_level: str = Field(..., description="学歴")
    goals: str = Field(..., description="目標")
    challenges: str = Field(..., description="課題")

# コンテンツアイデア
class ContentIdea(BaseModel):
    phase: str = Field(..., description="フェーズ")
    content_type: str = Field(..., description="コンテンツ形式")
    content_title: str = Field(..., description="コンテンツタイトル")
    content_overview: str = Field(..., description="コンテンツ概要")

class ContentIdeaList(BaseModel):
    content_idea_list: Optional[List[ContentIdea]] = Field(..., description="コンテンツアイデアのリスト")

# エージェント状態（マーケティングプロセス用）
class AgentState(TypedDict):
    plan: List[Plan]              # 各タスクの計画内容
    user_request: str             # ユーザーからのリクエスト
    existing_tasks: List[Task]      # 既存のタスク一覧（JIRAから取得）
    tasks: List[Task]             # 新規作成されたタスク一覧
    feedback: Optional[Dict[str, Feedback]]  # タスクごとのフィードバック
    complete: bool                # すべてのタスクが完了したか
    iteration: int                # 現在のイテレーション数

# エージェント実行結果（最終状態）
AgentResult = AgentState
