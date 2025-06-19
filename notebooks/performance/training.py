import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import Trainer, TrainingArguments


class KDTrainingArguments(TrainingArguments):
    """Training arguments for knowledge distillation."""
    def __init__(self, *args, alpha=0.5, temperature=2.0, **kwargs):
        """Constructor.

        Args:
            alpha (float, optional): Controls the balance between CE-loss and KL-loss. Defaults to 0.5.
            temperature (float, optional): Temperature of the softmax function. Defaults to 2.0.
        """
        # call the parent constructor
        super().__init__(*args, **kwargs)
        # save hyperparameters
        self.alpha = alpha
        self.temperature = temperature


class KDTrainer(Trainer):
    """Trainer for knowledge distillation.
    
    Loss Function: alpha * CE-loss + (1 - alpha) * T^2 * KLD-Loss
    """
    def __init__(self, *args, teacher_model: nn.Module, **kwargs):
        """Constructor.

        Args:
            teacher_model (torch.nn.Module): Teacher model.
        """
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model

    def compute_loss(self, model, inputs, num_items_in_batch, return_outputs=False):
        """Compute the distillation loss

        Args:
            model (nn.Module): Student model
            inputs (_type_): Student model inputs
            return_outputs (bool, optional): True if outputs should be returned. Defaults to False.

        Returns:
            _type_: Model outputs
        """
        # run inputs through student model
        outputs_student = model(**inputs)
        # extract CE-loss and logits from student
        loss_ce = outputs_student.loss
        logits_student = outputs_student.logits
        # extract logits from teacher
        with torch.no_grad():
            outputs_teacher = self.teacher_model(**inputs)
            logits_teacher = outputs_teacher.logits
        # soften probabilities and compute distillation loss
        loss_func = nn.KLDivLoss(reduction="batchmean")
        loss_kld = self.args.temperature ** 2 * loss_func(
            F.log_softmax(logits_student / self.args.temperature, dim=-1),
            F.softmax(logits_teacher / self.args.temperature, dim=-1)
        )
        # return weighted student loss
        loss = self.args.alpha * loss_ce + (1. - self.args.alpha) * loss_kld
        return (loss, outputs_student) if return_outputs else loss

    
