from pydantic import BaseModel, Field


# class Task(BaseModel):
#     hypothesis: str = Field(description="分析レポートにおいて検証可能な仮説")
#     purpose: str = Field(description="仮説の検証目的")
#     description: str = Field(description="具体的な分析方針と可視化対象")
#     chart_type: str = Field(description="グラフ想定, 例: ヒストグラム、棒グラフなど")


# class Plan(BaseModel):
#     purpose: str = Field(description="タスク要求から解釈される問い合わせ目的")
#     archivement: str = Field(description="タスク要求から推測されるタスク達成条件")
#     tasks: list[Task]


class Task(BaseModel):
    hypothesis: str = Field(
        title="仮説",
        description=(
            "検証可能な仮説を、その推測理由とともに詳細に記述する。"
            "仮説は、データ分析によって検証したい因果関係や傾向、または期待される結果について、"
            "具体的かつ明確に表現する。"
        ),
        examples=[
            "週末は多くの人が買い物をするため、土日は売上が増加する。",
            "新規ユーザーへのプロモーション施策が初回購入率を向上させる。",
        ],
    )
    purpose: str = Field(
        title="仮説の検証目的",
        description=(
            "この仮説を検証することで明らかにしたい課題や目的を具体的に記述する。"
            "仮説の検証がどのような意思決定や業務改善につながるか、またはどのような知見を得たいのかを明確に示す。"
        ),
        examples=[
            "曜日ごとの売上の違いを検証する。",
            "プロモーション施策の購買行動の変化についてターゲット層別にその影響を明らかにする。",
        ],
    )
    description: str = Field(
        title="分析手法の詳細",
        description=(
            "どのような分析手法（例：単変量解析、多変量解析、回帰分析、クラスタリングなど）を用いるか記述する。"
            "どの変数を使用するか、また関数の引数・戻り値を指定し、どのような比較や可視化を行うか詳細に記述する。"
        ),
    )
    chart_type: str = Field(
        title="グラフ想定",
        description="想定する可視化の種類を記述する。",
        examples=[
            "ヒストグラム",
            "棒グラフ",
            "折れ線グラフ",
            "円グラフ",
            "散布図",
        ],
    )


# class Plan(BaseModel):
#     purpose: str = Field(description="タスク要求から解釈される問い合わせ目的")
#     archivement: str = Field(description="タスク要求から推測されるタスク達成条件")
#     tasks: list[Task]


class Plan(BaseModel):
    purpose: str = Field(description="タスク要求から解釈される問い合わせ目的")
    archivement: str = Field(description="タスク要求から推測されるタスク達成条件")
    tasks: list[Task]
