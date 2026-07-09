import os
import json
from typing import Iterator, Tuple
from abc import ABC, abstractmethod

class BaseDataLoader(ABC):
    """
    Abstract Base Class for evaluation data loaders.
    Any custom data loader should inherit from this class and implement __iter__.
    """
    @abstractmethod
    def __iter__(self) -> Iterator[Tuple[str, str]]:
        """
        Yields:
            Tuple[str, str]: (question, gold_sparql)
        """
        pass

    @abstractmethod
    def __len__(self) -> int:
        """
        Returns:
            int: The total number of test cases.
        """
        pass


class Qald10DataLoader(BaseDataLoader):
    """
    Data loader for the QALD-10 dataset.
    Loads json file from qald_10.json and yields tuples of (question, sparql_query) for English queries.
    """
    def __init__(self, filepath: str = None):
        if filepath is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            filepath = os.path.join(base_dir, "data", "QALD-10-main", "data", "qald_10", "qald_10.json")
        
        self.filepath = filepath
        self.test_cases = []
        self._load_data()

    def _load_data(self):
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"QALD-10 dataset file not found at {self.filepath}")

        with open(self.filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        questions = data.get("questions", [])
        for item in questions:
            # Find the English version of the question
            question_str = None
            q_list = item.get("question", [])
            for q_info in q_list:
                if q_info.get("language") == "en":
                    question_str = q_info.get("string")
                    break

            sparql_query = item.get("query", {}).get("sparql")

            if question_str and sparql_query:
                self.test_cases.append((question_str.strip(), sparql_query.strip()))

    def __iter__(self) -> Iterator[Tuple[str, str]]:
        for question, sparql_query in self.test_cases:
            yield question, sparql_query

    def __len__(self) -> int:
        return len(self.test_cases)
