from typing import Optional, Any, Dict, List
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule
from transformers import T5Tokenizer
from datasets import load_dataset


class SummarizationDataLoader(LightningDataModule):
    """
    DataLoader class for loading and processing datasets for text summarization using T5 model.

    Attributes:
        dataset_name (str): The name of the dataset to load.
        tokenizer (T5Tokenizer): The tokenizer to use for encoding the text.
        batch_size (int): The batch size for the dataloaders.
        num_workers (int): The number of workers for the dataloaders.
    """

    def __init__(self, dataset_name: str = 'FiscalNote/billsum',
                 tokenizer_name: str = 't5-small', batch_size: int = 4, num_workers: int = 4):
        """
        Initializes the DataLoader with the specified dataset, tokenizer, and batch size.

        Args:
            dataset_name (str): The name of the dataset to load.
            tokenizer_name (str): The name of the tokenizer to use.
            batch_size (int): The batch size for the dataloaders.
            num_workers (int): The number of workers for the dataloaders.
        """
        super().__init__()
        self.dataset_name = dataset_name
        self.tokenizer = T5Tokenizer.from_pretrained(tokenizer_name)
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self) -> None:
        """
        Loads the dataset specified by the dataset_name attribute.
        """
        try:
            self.dataset = load_dataset(self.dataset_name)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load dataset {self.dataset_name}: {e}")

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Sets up the train, validation, and test datasets.

        Args:
            stage (Optional[str]): The stage of the setup (e.g., 'fit', 'validate', 'test', 'predict').
        """
        self.train_dataset = self.dataset['train']
        self.val_dataset = self.dataset['ca_test']
        self.test_dataset = self.dataset['test']

    def train_dataloader(self) -> DataLoader:
        """
        Returns the DataLoader for the training dataset.

        Returns:
            DataLoader: The DataLoader for the training dataset.
        """
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                          collate_fn=self.collate_fn, num_workers=self.num_workers)

    def val_dataloader(self) -> DataLoader:
        """
        Returns the DataLoader for the validation dataset.

        Returns:
            DataLoader: The DataLoader for the validation dataset.
        """
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                          collate_fn=self.collate_fn, num_workers=self.num_workers)

    def test_dataloader(self) -> DataLoader:
        """
        Returns the DataLoader for the test dataset.

        Returns:
            DataLoader: The DataLoader for the test dataset.
        """
        return DataLoader(self.test_dataset, batch_size=self.batch_size,
                          collate_fn=self.collate_fn, num_workers=self.num_workers)

    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Collates a batch of data, encoding the text and summary using the tokenizer.

        Args:
            batch (List[Dict[str, Any]]): A batch of data.

        Returns:
            Dict[str, Any]: The collated batch with encoded text and summary.
        """
        text = [item['text'] for item in batch]
        summary = [item['summary'] for item in batch]

        # Encode the text
        encodings = self.tokenizer(
            text_target=text, max_length=1024, padding='max_length', truncation=True, return_tensors='pt'
        )

        # Encode the summary
        labels = self.tokenizer(
            text_target=summary, max_length=128, padding='max_length', truncation=True, return_tensors='pt'
        ).input_ids

        labels[labels == 0] = -100  # To ignore pad tokens in loss computation
        return dict(input_ids=encodings.input_ids, attention_mask=encodings.attention_mask, labels=labels)
