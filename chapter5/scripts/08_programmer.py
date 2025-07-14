from loguru import logger

from scripts.programmer import programmer_node


def main() -> None:
    data_threads = programmer_node(
        data_file="data/sample.csv",
        user_request="データ概要について教えて",
        process_id="08_programmer",
    )
    logger.info(data_threads)


if __name__ == "__main__":
    main()
