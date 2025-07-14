
from loguru import logger

from src.modules.describe_dataframe import describe_dataframe
from src.modules.generate_plan import generate_plan


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
