from dataset import HFCocoDetection, collate_fn_coco
from transformers import (
    AutoModelForObjectDetection,
    DeformableDetrImageProcessor,
    Trainer,
    TrainingArguments,
)


class TrainingCommand:
    """
    Train an object detection model using Hugging Face Trainer API.

    Args passed via __init__:
      - model_checkpoint: pretrained DETR checkpoint
      - train_img_dir, train_ann_file
      - val_img_dir, val_ann_file
      - training_args
    """
    def __init__(
        self,
        config,
        model_checkpoint: str,
        train_img_dir: str,
        train_ann_file: str,
        val_img_dir: str,
        val_ann_file: str,
        training_args: TrainingArguments,
    ):
        self.config = config
        self.model_checkpoint = model_checkpoint
        self.train_img_dir = train_img_dir
        self.train_ann_file = train_ann_file
        self.val_img_dir = val_img_dir
        self.val_ann_file = val_ann_file
        self.training_args = training_args

    def invoke(self):
        processor = DeformableDetrImageProcessor.from_pretrained(
            self.model_checkpoint,
            format="coco_detection"        # tells the processor to expect COCO boxes
        )
        model = AutoModelForObjectDetection.from_pretrained(pretrained_model_name_or_path=self.model_checkpoint, config=self.config)

        #train_ds = CocoDetection(root=self.train_img_dir, annFile=self.train_ann_file)
        #val_ds = CocoDetection(root=self.val_img_dir, annFile=self.val_ann_file)
        #train_ds = COCODetectionDataset(self.train_img_dir, self.train_ann_file)
        #val_ds = COCODetectionDataset(self.val_img_dir, self.val_ann_file)
        train_ds = HFCocoDetection(img_dir=self.train_img_dir, ann_file=self.train_ann_file, processor=processor)
        val_ds = HFCocoDetection(img_dir=self.val_img_dir, ann_file=self.val_ann_file, processor=processor)

        trainer = Trainer(
            model=model,
            args=self.training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            data_collator=collate_fn_coco,
            tokenizer=processor,
        )
        trainer.train()
        trainer.save_model(self.training_args.output_dir)
        processor.save_pretrained(self.training_args.output_dir)