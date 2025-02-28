# Core Functions 📡

## 1️⃣ Get Available Models
```python
models = vespa.list_models()
print(models)
```

## 2️⃣ Define Task Type, Select Dataset Format, and Train Model
```python
vespa.set_task("classification")
vespa.set_dataset_format("yolo")
vespa.set_model("resnet50")
result = vespa.train("path/to/dataset")
print(result)
```

Check [Request Formatting](request.md) for more details.