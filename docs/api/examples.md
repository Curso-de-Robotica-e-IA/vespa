# API Usage Examples 🎯

## Python
```python
import vespa

vespa.set_task("classification")
vespa.set_dataset_format("yolo")
vespa.set_model("resnet50")
result = vespa.train("path/to/dataset")
print(result)
```

## Batch Processing
```python
images = ["img1.jpg", "img2.jpg"]
vespa.set_task("regression")
vespa.set_dataset_format("coco")
vespa.set_model("retinanet")
results = vespa.batch_predict(image_paths=images)
print(results)
```