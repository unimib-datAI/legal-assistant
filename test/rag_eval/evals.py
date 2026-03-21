from typing import Any

from ragas import Dataset, DataTable


class RagasEvaluator:
    def __init__(self):
        pass

    def _load_dataset(self, dataset_name:str , dataset_file_ext: str, dataset_file_path: str) -> DataTable[Any]:
        return Dataset.load(name= dataset_name, file_ext=dataset_file_ext, file_path=dataset_file_path)

