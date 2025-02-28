# COCO Dataset Format 📌

The **COCO format** is a JSON-based annotation format widely used in computer vision.

## COCO File Structure
- **images/** - Folder containing images.
- **annotations.json** - A JSON file storing the dataset annotations.

Example annotation:
```json
{
  "images": [{"id": 1, "file_name": "image1.jpg"}],
  "annotations": [{
    "image_id": 1,
    "bbox": [x, y, width, height],
    "category_id": 1
  }],
  "categories": [{"id": 1, "name": "cat"}]
}
```

## Loading COCO Data in Vespa
```python
from vespa.datasets import coco_dataset

dataset = coco_dataset("path/to/coco/dataset")
```