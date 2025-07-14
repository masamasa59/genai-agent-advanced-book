
from loguru import logger

from src.modules import describe_dataframe, generate_plan


data_path = "data/sample.csv"
data_info = describe_dataframe(data_path)
user_request = "scoreを最大化するための広告キャンペーンを検討したい"

response = generate_plan(
    data_info=data_info,
    user_request=user_request,
    model="gpt-4o-mini-2024-07-18",
)
plan = response.content
logger.info(plan.model_dump())
