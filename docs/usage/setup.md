# Usage Guide 🚀

This guide explains how to use the Vespa library for various tasks, including classification, regression, and detection.

---

## 📌 Setup
### Installing Vespa
To install Vespa, use:
```bash
pip install vespa
```

### Verifying Installation
Check the installed version:
```bash
python -c "import vespa; print(vespa.__version__)"
```

### Importing Vespa
```python
import vespa
```

---

## 📌 Setting Up a Task
Before using the library, you need to define the type of task you are performing.
```python
vespa.set_task("classification")  # Options: "classification", "regression", "detection"
```

### Selecting a Dataset Format
```python
vespa.set_dataset_format("yolo")  # Options: "yolo", "coco", "pascal_voc"
```

### Choosing a Model
```python
vespa.set_model("resnet50")  # Options depend on the task type
```