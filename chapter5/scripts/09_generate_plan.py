import io

from loguru import logger

from src.modules import describe_dataframe, generate_plan


def main() -> None:
    data_path = "data/sample.csv"
    template_file = "src/prompts/generate_plan.jinja"
    user_request = "scoreを最大化するための広告キャンペーンを検討したい"

    with open(data_path, "rb") as fi:
        file_object = io.BytesIO(fi.read())
    data_info = describe_dataframe(file_object=file_object, template_file=template_file)

    response = generate_plan(
        data_info=data_info,
        user_request=user_request,
        model="gpt-4o-mini-2024-07-18",
    )
    plan = response.content
    logger.info(plan.model_dump_json(indent=4))


if __name__ == "__main__":
    main()
