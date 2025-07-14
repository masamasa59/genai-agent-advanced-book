from pydantic import BaseModel, Field


class Task(BaseModel):
    hypothesis: str = Field(description="分析レポートにおいて検証可能な仮説")
    purpose: str = Field(description="仮説の検証目的")
    description: str = Field(description="具体的な分析方針と可視化対象")
    chart_type: str = Field(description="グラフ想定（例：ヒストグラム、棒グラフなど）")


class Plan(BaseModel):
    purpose: str = Field(description="タスク要求から解釈される問い合わせ目的")
    archivement: str = Field(description="タスク要求から推測されるタスク達成条件")
    tasks: list[Task]
