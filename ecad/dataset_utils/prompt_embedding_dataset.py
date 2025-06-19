from typing import Generic
import torch
from torch.utils.data import Dataset
from pathlib import Path

from ecad.types import PromptEmbeddingType


class PromptEmbeddingDataset(Dataset, Generic[PromptEmbeddingType]):
    def __init__(self, embedding_dir):
        """
        Initializes the dataset by loading all embedding and attention mask tensors.

        Args:
            embedding_dir (str or Path): Directory containing the .pt files with embeddings.
        """
        self.embedding_dir = Path(embedding_dir)

        self.filenames = list(self.embedding_dir.glob("**/*.pt"))

    def _get_relative_path(self, idx):
        filepath = self.filenames[idx]
        rel_path = filepath.relative_to(self.embedding_dir)
        # of the form foo/../bar/baz.pt. want foo/.../bar/
        return str(rel_path.parent)

    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.
        """
        return len(self.filenames)

    def __getitem__(self, idx) -> dict[str, torch.Tensor | str]:
        """
        Retrieves a single sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            dict: A dictionary containing prompt embeddings, attention masks, and their negative counterparts.
        """

        loaded_dict = torch.load(
            self.filenames[idx], weights_only=True, map_location="cpu"
        )

        return_dict = {
            "name": self.filenames[idx].stem,
            "relative_path": self._get_relative_path(idx),
        }
        for key, value in loaded_dict.items():
            if value is None:
                continue
            elif isinstance(value, torch.Tensor):
                return_dict[key] = value.squeeze()
            else:
                return_dict[key] = value

        return return_dict
