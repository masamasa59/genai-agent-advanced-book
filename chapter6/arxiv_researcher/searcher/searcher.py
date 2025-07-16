from abc import ABC, abstractmethod
from typing import Any


class Searcher(ABC):
    @abstractmethod
    def run(self, goal_setting: str, query: str) -> list[Any]:
        pass
