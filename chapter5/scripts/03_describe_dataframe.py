import io

import pandas as pd
from loguru import logger

from src.llms.utils import load_template


def describe_dataframe(
    data_file: str,
    template_file: str,
) -> str:
    df = pd.read_csv(data_file)
    buf = io.StringIO()
    df.info(buf=buf)
    df_info = buf.getvalue()
    template = load_template(template_file)
    return template.render(
        df_info=df_info,
        df_sample=df.sample(5).to_markdown(),
        df_describe=df.describe().to_markdown(),
    )


def main() -> None:
    data_file = "data/sample.csv"
    template_file = "src/prompts/describe_dataframe.jinja"
    data_info = describe_dataframe(data_file=data_file, template_file=template_file)
    logger.info(data_info)


if __name__ == "__main__":
    main()
