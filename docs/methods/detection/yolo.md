# YOLO Model ⚡

## Overview
YOLO (You Only Look Once) is a **real-time object detection model** that processes images in a single pass.

## How to Use YOLO in Vespa
```python
import vespa
vespa.set_task("detection")
vespa.set_dataset_format("yolo")
vespa.set_model("yolo")
result = vespa.train("path/to/dataset")
print(result)
```

For more details, check the official [documentation](../index.md).