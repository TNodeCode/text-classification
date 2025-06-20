import os
from typing import Dict, List, Tuple

import torch
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from transformers import DetrForObjectDetection, DetrImageProcessor


class PyTorchEvaluationCommand:
    """
    Evaluate a PyTorch DETR model on COCO-format splits.
    """
    def __init__(
        self,
        model_dir: str,
        processor_dir: str,
        splits: dict[str, Tuple[str, str]],  # name -> (img_dir, ann_file)
        device: torch.device,
    ):
        self.model_dir = model_dir
        self.processor_dir = processor_dir
        self.splits = splits
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def invoke(self) -> Dict[str, List[float]]:
        processor = DetrImageProcessor.from_pretrained(self.processor_dir)
        model = DetrForObjectDetection.from_pretrained(self.model_dir).to(self.device).eval()
        results: Dict[str, List[float]] = {}
        for split, (img_dir, ann_file) in self.splits.items():
            coco = COCO(ann_file)
            preds = []
            for img_info in coco.loadImgs(coco.getImgIds()):
                img_id = img_info['id']
                image = Image.open(os.path.join(img_dir, img_info['file_name'])).convert('RGB')
                enc = processor(images=[image], return_tensors='pt')
                pix = enc['pixel_values'].to(self.device)
                with torch.no_grad():
                    out = model(pix)
                # post-process
                det = processor.post_process_object_detection(
                    out,
                    threshold=0.5,
                    target_sizes=[image.size[::-1]]
                )[0]
                for d in det:
                    preds.append({
                        'image_id': img_id,
                        'category_id': d['label_id'],
                        'bbox': d['box'],
                        'score': d['score']
                    })
            coco_pred = coco.loadRes(preds)
            coco_eval = COCOeval(coco, coco_pred, 'bbox')
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            results[split] = coco_eval.stats.tolist()
        return results