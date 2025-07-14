from pydantic import BaseModel, Field


class Review(BaseModel):
    observation: str = Field(description="コードに対するフィードバックやコメント")
    is_completed: bool = Field(description="実行結果がタスク要求を満たすか")
