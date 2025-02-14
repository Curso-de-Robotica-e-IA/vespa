"""
Detection Datasets Module

This module provides dataset loaders for object detection models in various formats, including YOLO, COCO, and Pascal VOC.

### **Usage:**
#### **Load a dataset:**
```python
from vespa.datasets.detection import load_dataset
dataset = load_dataset("path/to/dataset", "yolo", txt_file="train.txt", image_size=640, transforms=None, model_name="yolo_v5")
```

#### **Available Datasets:**
- **YOLO**
- **COCO**
- **Pascal VOC**

### **Functions:**
- **`load_dataset(dataset_path: str, dataset_format: str, **kwargs) -> BaseDataset`**  
  Loads a dataset in the specified format.

---

### **Author:**
- **Vespa AI Team**
"""

from .yolo.yolo_dataset import YOLODataset
from .coco.coco_dataset import COCODataset
from .pascal_voc.pascal_voc_dataset import PascalVOCDataset

def load_dataset(dataset_path: str, dataset_format: str, model_name, transforms, **kwargs):
    """
    Load the dataset based on the specified format.

    Args:
        dataset_path (str): Path to the dataset.
        dataset_format (str): Format of the dataset ('yolo', 'coco', 'pascal_voc').
        **kwargs: Additional arguments depending on the dataset format.

    Returns:
        BaseDataset: An instance of the dataset class.

    Raises:
        ValueError: If an unsupported dataset format is provided.
    """
    if dataset_format == "yolo":
        required_keys = ["txt_file"]
        for key in required_keys:
            if key not in kwargs:
                raise ValueError(f"Missing required argument '{key}' for YOLODataset.")

        return YOLODataset(
            root_dir=dataset_path,
            txt_file=kwargs["txt_file"],
            model_name=model_name,
            transforms=transforms,
        )

    elif dataset_format == "coco":
        required_keys = ["txt_file"]
        if "txt_file" not in kwargs:
            raise ValueError("Missing required argument 'txt_file' for COCODataset.")

        return COCODataset(
            root_dir=dataset_path,
            txt_file=kwargs["txt_file"],
            model_name=model_name,
            transforms=transforms
        )

    elif dataset_format == "pascal_voc":
        return PascalVOCDataset(
            root_dir=dataset_path,
            model_name=model_name,
            transforms=transforms,
        )

    else:
        raise ValueError(f"Unsupported dataset format: {dataset_format}")

__all__ = ["YOLO Dataset Format", "COCO Dataset Format", "PascalVOC Dataset Format", "load_dataset"]
