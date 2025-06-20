import os
from glob import glob

from PIL import Image
from torch.utils.data import Dataset


class ClassificationDataset(Dataset):
    """A simple dataset for image classification. Expects a root directory with sub-folders for each class."""
    def __init__(self, root_dir: str):
        """Constructor.

        Args:
            root_dir (str): Images root directory.
        """
        self.root_dir = root_dir
        # infer class names from subdirectory names
        self.classes = sorted([d.name for d in os.scandir(root_dir) if d.is_dir()])
        # disctionaries that map class names to class IDs and vice versa
        self.label2id = {cls: idx for idx, cls in enumerate(self.classes)}
        self.id2label = {idx: cls for cls, idx in self.label2id.items()}
        # gather image paths and labels
        self.image_paths: list[str] = []
        self.labels: list[int] = []
        for cls in self.classes:
            cls_dir = os.path.join(root_dir, cls)
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):  # supported formats
                for path in glob(os.path.join(cls_dir, ext)):
                    self.image_paths.append(path)
                    self.labels.append(self.label2id[cls])

    def __len__(self):
        """Get nunber of images in the dataset.

        Returns:
            int: Number of images in the dataset
        """
        return len(self.image_paths)

    def __getitem__(self, idx):
        """Load single image by an index

        Args:
            idx (int): Image index

        Returns:
            dict: image data
        """
        image = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]
        return {"image": image, "label": label}