from dotenv import load_dotenv
from e2b_code_interpreter import Sandbox
from loguru import logger

from src.modules.describe_dataframe import describe_dataframe
from src.modules.execute_code import execute_code
from src.modules.generate_review import generate_review
from src.modules.set_dataframe import set_dataframe


load_dotenv()

process_id = "07_generate_review"
data_path = "data/sample.csv"
data_info = describe_dataframe(data_path)

user_request = "データフレームのサイズを確認する"

with Sandbox() as sandbox:
    set_dataframe(sandbox, data_path)
    data_thread = execute_code(
        process_id=process_id,
        thread_id=0,
        sandbox=sandbox,
        user_request=user_request,
        code="print(df.shape)",
    )
    logger.info(data_thread.model_dump())

response = generate_review(
    user_request=user_request,
    data_info=data_info,
    data_thread=data_thread,
)
review = response.content
print(review.model_dump())
