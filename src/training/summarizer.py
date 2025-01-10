from typing import Optional, Tuple, Dict, Any
from pytorch_lightning import LightningModule
from transformers import T5ForConditionalGeneration, T5Tokenizer, AdamW
from torch import Tensor
import evaluate


class Summarizer(LightningModule):
    """
    LightningModule for training a T5 model for text summarization.

    Attributes:
        model (T5ForConditionalGeneration): The T5 model for conditional generation.
        lr (float): The learning rate for the optimizer.
        rouge: The ROUGE score evaluator.
    """

    def __init__(self, model_name: str = 't5-small', lr: float = 1e-4):
        """
        Initializes the Summarizer with the specified model and learning rate.

        Args:
            model_name (str): The name of the pre-trained T5 model to use.
            lr (float): The learning rate for the optimizer.
        """
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(
            model_name)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.lr = lr
        self.rouge = evaluate.load('rouge')

    def forward(self, input_ids: Tensor, attention_mask: Tensor,
                labels: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """
        Forward pass through the model.

        Args:
            input_ids (Tensor): The input IDs for the model.
            attention_mask (Tensor): The attention mask for the input IDs.
            labels (Optional[Tensor]): The labels for the input IDs.

        Returns:
            Tuple[Tensor, Tensor]: The loss and logits from the model.
        """
        # Pass the inputs through the model
        output = self.model(
            input_ids, attention_mask=attention_mask, labels=labels)
        # Return the loss and logits
        return output.loss, output.logits

    def training_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Tensor:
        """
        Training step for the model.

        Args:
            batch (Dict[str, Tensor]): A batch of data containing input IDs, attention masks, and labels.
            batch_idx (int): The index of the batch.

        Returns:
            Tensor: The loss for the training step.
        """
        # Extract input IDs, attention masks, and labels from the batch
        input_ids, attention_mask, labels = batch['input_ids'], batch['attention_mask'], batch['labels']
        # Perform a forward pass and compute the loss
        loss, _ = self(input_ids, attention_mask, labels)
        # Log the training loss
        self.log('train_loss', loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Dict[str, Any]:
        """
        Validation step for the model.

        Args:
            batch (Dict[str, Tensor]): A batch of data containing input IDs, attention masks, and labels.
            batch_idx (int): The index of the batch.

        Returns:
            Dict[str, Any]: The loss and ROUGE scores for the validation step.
        """
        # Extract input IDs, attention masks, and labels from the batch
        input_ids, attention_mask, labels = batch['input_ids'], batch['attention_mask'], batch['labels']
        # Perform a forward pass and compute the loss
        loss, logits = self(input_ids, attention_mask, labels)
        # Decode the predictions and labels
        preds = self.model.generate(input_ids, attention_mask=attention_mask)
        decoded_preds = self.tokenizer.batch_decode(
            preds, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(
            labels, skip_special_tokens=True)
        # Compute ROUGE scores
        rouge_scores = self.rouge.compute(
            predictions=decoded_preds, references=decoded_labels)
        # Log the validation loss and ROUGE scores
        self.log('val_loss', loss, prog_bar=True, logger=True)
        self.log('rouge1', rouge_scores['rouge1'], prog_bar=True, logger=True)
        self.log('rouge2', rouge_scores['rouge2'], prog_bar=True, logger=True)
        self.log('rougeL', rouge_scores['rougeL'], prog_bar=True, logger=True)
        return {'val_loss': loss, 'rouge_scores': rouge_scores}

    def configure_optimizers(self) -> AdamW:
        """
        Configures the optimizer for the model.

        Returns:
            AdamW: The optimizer for the model.
        """
        # Create and return the AdamW optimizer with the specified learning rate
        return AdamW(self.parameters(), lr=self.lr)
