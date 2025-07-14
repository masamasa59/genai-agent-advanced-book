from pydantic import BaseModel, Field


class Program(BaseModel):
    achievement_condition: str = Field(description="要求の達成条件")
    execution_plan: str = Field(description="実行計画")
    code: str = Field(description="生成対象となるコード")
