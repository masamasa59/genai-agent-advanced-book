# テンプレートエンジン Jinja2 の使い方（テキストから使用する場合）
from jinja2 import Template


def main() -> None:
    # render の入力引数に message が指定された場合にそれを表示する穴埋め型テンプレート
    source = """{% if message %}メッセージがあります: {{ message }}{% endif %}"""
    template = Template(source=source)

    # 1.引数に message を指定した場合
    print("1.", template.render(message="hello"))
    # 2.引数に message を指定しなかった場合
    print("2.", template.render())


if __name__ == "__main__":
    main()
