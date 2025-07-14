import io

from dotenv import load_dotenv
from e2b_code_interpreter import Sandbox
from loguru import logger

from src.modules import (
    describe_dataframe,
    execute_code,
    generate_review,
    set_dataframe,
)


load_dotenv()


def main() -> None:
    process_id = "07_generate_review"
    data_path = "data/sample.csv"
    template_file = "src/prompts/generate_review.jinja"
    user_request = "データフレームのサイズを確認する"

    with open(data_path, "rb") as fi:
        file_object = io.BytesIO(fi.read())
    data_info = describe_dataframe(file_object=file_object, template_file=template_file)

    with Sandbox() as sandbox:
        with open(data_path, "rb") as fi:
            set_dataframe(sandbox=sandbox, file_object=fi)
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
    logger.info(review.model_dump_json(indent=4))


if __name__ == "__main__":
    main()
