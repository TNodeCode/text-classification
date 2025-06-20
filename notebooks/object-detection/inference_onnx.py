import os
from typing import Dict, List

import onnxruntime as ort
import torch
from PIL import Image
from transformers import DetrImageProcessor


class ONNXInferenceCommand:
    """
    Run object detection inference using ONNX Runtime and post-process with DetrImageProcessor.
    """
    def __init__(
        self,
        onnx_model_path: str,
        processor_dir: str,
        image_paths: list[str],
        output_dir: str,
        threshold: float = 0.5,
    ):
        self.onnx_model_path = onnx_model_path
        self.processor_dir = processor_dir
        self.image_paths = image_paths
        self.output_dir = output_dir
        self.threshold = threshold
        os.makedirs(self.output_dir, exist_ok=True)
        self.session = ort.InferenceSession(self.onnx_model_path, providers=["CPUExecutionProvider"])

    def invoke(self) -> Dict[str, List[Dict]]:
        processor = DetrImageProcessor.from_pretrained(self.processor_dir)
        results: Dict[str, List[Dict]] = {}
        for img_path in self.image_paths:
            image = Image.open(img_path).convert('RGB')
            enc = processor(images=[image], return_tensors='pt')
            pv = enc['pixel_values'].cpu().numpy()
            logits, boxes = self.session.run(None, {'pixel_values': pv})
            # Post-process
            detections = processor.post_process_object_detection(
                {'logits': torch.tensor(logits), 'pred_boxes': torch.tensor(boxes)},
                threshold=self.threshold,
                target_sizes=[image.size[::-1]]
            )[0]
            results[img_path] = detections
        return results