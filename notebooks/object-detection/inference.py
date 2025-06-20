import os
from typing import Dict, List

from PIL import Image
from transformers import DetrForObjectDetection, DetrImageProcessor, pipeline


class PyTorchInferenceCommand:
    """
    Run object detection inference using Hugging Face pipeline.
    """
    def __init__(
        self,
        model_dir: str,
        processor_dir: str,
        image_paths: list[str],
        output_dir: str,
        device: str,
    ):
        self.model_dir = model_dir
        self.processor_dir = processor_dir
        self.image_paths = image_paths
        self.output_dir = output_dir
        self.device = device
        os.makedirs(self.output_dir, exist_ok=True)

    def invoke(self) -> Dict[str, List[Dict]]:
        processor = DetrImageProcessor.from_pretrained(self.processor_dir)
        model = DetrForObjectDetection.from_pretrained(self.model_dir)
        det_pipe = pipeline(
            'object-detection',
            model=model,
            feature_extractor=processor,
            device=self.device
        )
        results: Dict[str, List[Dict]] = {}
        for img_path in self.image_paths:
            image = Image.open(img_path).convert('RGB')
            outputs = det_pipe(image)
            # outputs: list of {'score','label','box'}
            results[img_path] = outputs
            # optionally save boxes/labels
        return results