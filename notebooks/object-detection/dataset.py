
import torch
from torchvision.datasets import CocoDetection

    
class HFCocoDetection(CocoDetection):
    """Huggingface COCO Object Detection Dataset.

    Args:
        CocoDetection (torchvision.datasets.Dataset): COCO Object Detection Dataset
    """
    def __init__(self, img_dir, ann_file, processor):
        super().__init__(img_dir, ann_file)
        self.processor = processor

    def __getitem__(self, idx: int):
        """Get dataset item.

        Args:
            idx (int): Dataset item index

        Returns:
            dict: Dictionary with the following structure:
                  {
                    pixel_values: tensor of shape (C, H, W)
                    labels: {
                      size: tensor([H,W])
                      image_id: tensor([image_id])
                      class_labels: tensor([class1,...,classN])
                      boxes: tensor([box1,...,boxN])
                      area: tensor([area1,...,areaN])
                      iscrowd: tensor([iscrowd1,...,iscrowdN])
                      orig_size: tensor([H,W])
                    }
                  }
        """
        image, target = super().__getitem__(idx)

        # COCO → lists of boxes & labels
        annotations={"image_id": 0, "annotations": [{
            "image_id": target[0]["image_id"],
            "category_id": target[0]["category_id"],
            "iscrowd": target[0]["iscrowd"],
            "area": target[0]["area"],
            "bbox": target[0]["bbox"],
        }]}

        encoded = self.processor(
            images=image,
            annotations=annotations,
            return_tensors="pt",
        )

        # remove the added batch dim
        encoded = {k: v.squeeze(0) if type(v) is torch.Tensor else v for k, v in encoded.items()}

        return {
            # remove the artificial batch dim *only* from tensors
            "pixel_values": encoded["pixel_values"].squeeze(0),
            "labels":       encoded["labels"][0],
        }
    
def collate_fn_coco(batch):
    # batch is a list of dicts coming from __getitem__
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    labels       = [item["labels"] for item in batch]  # keep as list (variable length)
    return {"pixel_values": pixel_values, "labels": labels}

