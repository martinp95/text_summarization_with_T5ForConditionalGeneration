
from typing import Optional, Tuple, Dict
from pytorch_lightning import LightningModule
from transformers import T5ForConditionalGeneration, T5Tokenizer
from torch import Tensor
from torch.optim import AdamW


class Summarizer(LightningModule):
    """
    PyTorch LightningModule for training a T5 model for text summarization.

    This class encapsulates the model, forward pass, training, validation, testing,
    and optimizer configuration, making it easier to train and evaluate the model.

    Attributes:
        model (T5ForConditionalGeneration): The pre-trained T5 model for text summarization.
        tokenizer (T5Tokenizer): Tokenizer corresponding to the T5 model.
        lr (float): Learning rate for the AdamW optimizer.
    """

    def __init__(self, model_name: str = 't5-small', lr: float = 1e-4):
        """
        Initializes the Summarizer module with the specified pre-trained model and learning rate.

        Args:
            model_name (str): The name of the pre-trained T5 model to load (default: 't5-small').
            lr (float): The learning rate for the optimizer (default: 1e-4).
        """
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.lr = lr

    def forward(self, input_ids: Tensor, attention_mask: Tensor,
                labels: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """
        Defines the forward pass for the model.

        Args:
            input_ids (Tensor): Tokenized input sequences.
            attention_mask (Tensor): Mask to avoid performing attention on padding tokens.
            labels (Optional[Tensor]): Target sequences for training.

        Returns:
            Tuple[Tensor, Tensor]: A tuple containing the loss (if labels are provided) and logits.
        """
        # Pass the inputs through the model
        output = self.model(input_ids=input_ids,
                            attention_mask=attention_mask, labels=labels)
        # Return the loss and logits
        return output.loss, output.logits

    def training_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Tensor:
        """
        Defines the training step logic.

        Args:
            batch (Dict[str, Tensor]): Batch containing input IDs, attention masks, and labels.
            batch_idx (int): Index of the current batch.

        Returns:
            Tensor: Training loss for the batch.
        """
        # Extract input IDs, attention masks, and labels from the batch
        input_ids, attention_mask, labels = batch['input_ids'], batch['attention_mask'], batch['labels']

        # Perform a forward pass and compute the loss
        loss, _ = self(input_ids, attention_mask, labels)

        # Log the test loss
        self.log('train_loss', loss, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch: Dict[str, Tensor], batch_idx: int) -> None:
        """
        Defines the validation step logic.

        Args:
            batch (Dict[str, Tensor]): Batch containing input IDs, attention masks, and labels.
            batch_idx (int): Index of the current batch.
        """
        # Extract input IDs, attention masks, and labels from the batch
        input_ids, attention_mask, labels = batch['input_ids'], batch['attention_mask'], batch['labels']

        # Perform a forward pass and compute the loss
        loss, _ = self(input_ids, attention_mask, labels)

        # Log the test loss
        self.log('val_loss', loss, prog_bar=True, logger=True)

    def test_step(self, batch: Dict[str, Tensor], batch_idx: int) -> None:
        """
        Defines the test step logic.

        Args:
            batch (Dict[str, Tensor]): Batch containing input IDs, attention masks, and labels.
            batch_idx (int): Index of the current batch.
        """
        # Extract input IDs, attention masks, and labels from the batch
        input_ids, attention_mask, labels = batch['input_ids'], batch['attention_mask'], batch['labels']

        # Perform a forward pass and compute the loss
        loss, _ = self(input_ids, attention_mask, labels)

        # Log the test loss
        self.log('test_loss', loss, prog_bar=True, logger=True)

    def configure_optimizers(self) -> AdamW:
        """
        Configures the optimizer for training the model.

        Returns:
            AdamW: Optimizer initialized with model parameters and the specified learning rate.
        """
        return AdamW(self.parameters(), lr=self.lr)
