from typing import Optional, Any, Dict, List
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule
from transformers import T5Tokenizer
from datasets import load_dataset


class SummarizationDataLoader(LightningDataModule):
    """
    A PyTorch Lightning DataModule for preparing and loading datasets for text summarization tasks
    using the T5 model. This class encapsulates dataset loading, tokenization, and DataLoader creation.

    Attributes:
        dataset_name (str): Name of the dataset to load (default: 'FiscalNote/billsum').
        tokenizer (T5Tokenizer): Tokenizer for text processing.
        batch_size (int): Number of samples per batch.
        num_workers (int): Number of workers for DataLoader.
    """

    def __init__(self, dataset_name: str = 'FiscalNote/billsum',
                 tokenizer_name: str = 't5-small', batch_size: int = 4, num_workers: int = 4):
        """
        Initializes the DataLoader with the specified dataset, tokenizer, batch size, and number of workers.

        Args:
            dataset_name (str): The name of the dataset to load (default: 'FiscalNote/billsum').
            tokenizer_name (str): Pretrained tokenizer name or path (default: 't5-small').
            batch_size (int): Number of samples per batch (default: 4).
            num_workers (int): Number of workers for DataLoader (default: 4).
        """
        super().__init__()
        self.dataset_name = dataset_name
        self.tokenizer = T5Tokenizer.from_pretrained(tokenizer_name)
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self) -> None:
        """
        Loads the dataset specified by `dataset_name`.
        Raises:
            RuntimeError: If the dataset fails to load.
        """
        try:
            self.dataset = load_dataset(self.dataset_name)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load dataset {self.dataset_name}: {e}")

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Prepares the train, validation, and test datasets for use in DataLoaders.

        Args:
            stage (Optional[str]): Stage of the setup ('fit', 'validate', 'test', 'predict').
        """
        self.train_dataset = self.dataset['train']
        self.val_dataset = self.dataset['ca_test']
        self.test_dataset = self.dataset['test']

    def train_dataloader(self) -> DataLoader:
        """
        Creates the DataLoader for the training dataset.

        Returns:
            DataLoader: DataLoader instance for the training dataset.
        """
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size,
            collate_fn=self.collate_fn, num_workers=self.num_workers
        )

    def val_dataloader(self) -> DataLoader:
        """
        Creates the DataLoader for the validation dataset.

        Returns:
            DataLoader: DataLoader instance for the validation dataset.
        """
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size,
            collate_fn=self.collate_fn, num_workers=self.num_workers
        )

    def test_dataloader(self) -> DataLoader:
        """
        Creates the DataLoader for the test dataset.

        Returns:
            DataLoader: DataLoader instance for the test dataset.
        """
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size,
            collate_fn=self.collate_fn, num_workers=self.num_workers
        )

    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Tokenizes and encodes a batch of data for text summarization.

        Args:
            batch (List[Dict[str, Any]]): A batch of data, where each entry contains 'text' and 'summary' keys.

        Returns:
            Dict[str, Any]: A dictionary containing:
                - input_ids: Encoded input text tensor.
                - attention_mask: Attention mask tensor.
                - labels: Encoded summary tensor with padding tokens ignored in loss computation.
        """
        prefix = "summarize: "
        text = [prefix + item['text'] for item in batch]
        summary = [item['summary'] for item in batch]

        # Encode the input text
        encodings = self.tokenizer(
            text, max_length=1024, padding='max_length',
            truncation=True, return_tensors='pt'
        )

        # Encode the target summaries
        labels = self.tokenizer(
            summary, max_length=128, padding='max_length',
            truncation=True, return_tensors='pt'
        ).input_ids

        # Replace padding tokens with -100 to ignore them during loss computation
        labels[labels == 0] = -100

        return {
            "input_ids": encodings.input_ids,
            "attention_mask": encodings.attention_mask,
            "labels": labels
        }
