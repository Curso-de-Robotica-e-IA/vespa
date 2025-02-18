# RCNN Model 🦾

## Overview
Region-based Convolutional Neural Networks (RCNN) are a family of models used for **object detection**. It works by selecting **region proposals** and classifying them using a CNN.

## How to Use RCNN in Vespa
```python
import vespa
vespa.set_task("detection")
vespa.set_dataset_format("coco")
vespa.set_model("rcnn")
result = vespa.train("path/to/dataset")
print(result)
```