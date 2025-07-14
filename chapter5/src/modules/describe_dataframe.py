import io

import pandas as pd

from src.llms.utils import load_template


def describe_dataframe(
    file_object: io.BytesIO,
    template_file: str = "src/prompts/describe_dataframe.jinja",
) -> str:
    # CSVファイルを読み込み、データフレームを作成
    df = pd.read_csv(file_object)
    # データフレームの概要情報を取得
    buf = io.StringIO()
    df.info(buf=buf)
    df_info = buf.getvalue()
    # データフレーム情報を構築して返す
    template = load_template(template_file)
    return template.render(
        df_info=df_info,
        df_sample=df.sample(5).to_markdown(),
        df_describe=df.describe().to_markdown(),
    )
