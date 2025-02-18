# RetinaNet Model 🔎

## Overview
RetinaNet is a **single-stage object detection model** that uses **Focal Loss** to improve accuracy on small and hard-to-detect objects.

## How to Use RetinaNet in Vespa
```python
import vespa
vespa.set_task("detection")
vespa.set_dataset_format("pascal_voc")
vespa.set_model("retinanet")
result = vespa.train("path/to/dataset")
print(result)
```