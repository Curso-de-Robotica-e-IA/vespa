# API Requests 📤

## Function Parameters
### Image Classification
```python
vespa.set_task("classification")
vespa.set_dataset_format("yolo")
vespa.set_model("resnet50")
vespa.train("path/to/dataset")
```
### Batch Inference
```python
images = ["img1.jpg", "img2.jpg"]
vespa.set_task("regression")
vespa.set_dataset_format("coco")
vespa.set_model("retinanet")
results = vespa.batch_predict(image_paths=images)
```