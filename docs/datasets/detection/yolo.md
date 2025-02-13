# YOLO Dataset Format 📌

The **YOLO format** is commonly used for object detection models. It consists of:
- **Images** stored in a folder (e.g., `images/`)
- **Labels** stored in a folder (e.g., `labels/`)

## Loading YOLO Data in Vespa
```python
from vespa.datasets import yolo_dataset

dataset = yolo_dataset("path/to/yolo/dataset")
```