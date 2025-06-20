from typing import List

import torch
from dataset import ClassificationDataset
from transformers import (
    AutoImageProcessor,
    ResNetConfig,
    ResNetForImageClassification,
    Trainer,
    TrainingArguments,
)


class TrainingCommand:
    """
    Train an image classification model using Hugging Face Trainer API.
    """
    def __init__(
        self,
        model_checkpoint: str,
        train_dir: str,
        val_dir: str,
        classes: List[str],
        training_args: TrainingArguments,
    ):
        self.model_checkpoint = model_checkpoint
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.classes = classes
        self.training_args = training_args

    def invoke(self):
        # prepare config, processor, model
        num_labels = len(self.classes)
        label2id = {cls: i for i, cls in enumerate(self.classes)}
        id2label = {i: cls for cls, i in label2id.items()}
        config = ResNetConfig.from_pretrained(
            self.model_checkpoint,
            num_labels=num_labels,
            label2id=label2id,
            id2label=id2label,
        )
        processor = AutoImageProcessor.from_pretrained(self.model_checkpoint)
        model = ResNetForImageClassification.from_pretrained(
            self.model_checkpoint,
            config=config,
            ignore_mismatched_sizes=True,
        )

        # build datasets
        train_ds = ClassificationDataset(self.train_dir)
        val_ds = ClassificationDataset(self.val_dir)

        # collate_fn
        def collate_fn(batch):
            images = [item['image'] for item in batch]
            labels = [item['label'] for item in batch]
            enc = processor(images=images, return_tensors='pt')
            enc['labels'] = torch.tensor(labels, dtype=torch.long)
            return enc

        trainer = Trainer(
            model=model,
            args=self.training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            data_collator=collate_fn,
            tokenizer=processor,  # moves pixel_values to device
        )
        trainer.train()
        trainer.save_model(self.training_args.output_dir)
        processor.save_pretrained(self.training_args.output_dir)