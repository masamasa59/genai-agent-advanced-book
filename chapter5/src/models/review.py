from pydantic import BaseModel, Field


# class Review(BaseModel):
#     observation: str = Field(description="コードに対するフィードバックやコメント")
#     is_completed: bool = Field(description="実行結果がタスク要求を満たすか")


class Review(BaseModel):
    observation: str = Field(
        title="実行結果に対する客観的受け止め",
        description=(
            "まずはコードの実行結果に対する客観的な事実を記述する。例えば「正常に終了し、〇〇という結果を得た。」「エラーが発生した。」などを記述する。"
            "その後、コードの実行結果がユーザーから与えられた要求に対して最低限担保できているかを評価する。"
            "要求を満たさない場合は、その修正方針を追記する。"
        ),
    )
    is_completed: bool = Field(
        title="タスク達成条件",
        description=(
            "実行結果がユーザーから与えられた要求に対して最低限担保できているかを評価する。"
            "タスク要求を満たさない場合はFalse、改善点はあれど最低限要求を満たす場合はTrueとする。"
        ),
    )
