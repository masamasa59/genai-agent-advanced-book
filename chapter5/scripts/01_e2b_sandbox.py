# プログラムリスト5.3: E2B Sandbox の使い方
from dotenv import load_dotenv
from e2b_code_interpreter import Sandbox
from loguru import logger


def main() -> None:
    load_dotenv()
    with Sandbox() as sandbox:
        execution = sandbox.run_code("print('Hello World!')")
    logger.info("\n".join(execution.logs.stdout))


if __name__ == "__main__":
    main()
