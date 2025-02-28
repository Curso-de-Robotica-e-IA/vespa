# Development Guide 🚀

This section provides guidelines for contributing to the Vespa library, understanding its structure, and setting up a development environment.

## 📁 Project Structure
```
vespa/
├── datasets/           # Dataset handling and transformations
│   ├── detection/
│   │   ├── coco/
│   │   │   ├── coco_transforms.py
│   │   ├── pascal_voc/
│   │   │   ├── pascal_voc_dataset.py
│   │   │   ├── pascal_voc_transforms.py
│   │   ├── yolo/
│   │   │   ├── yolo_dataset.py
│   │   │   ├── yolo_transforms.py
│   ├── base_dataset.py
│   ├── config.py
│   ├── hypes.yaml
│   ├── utils.py
│
├── methods/            # Implementation of different ML methods
│   ├── detection/
│   │   ├── onnx_yolo/
│   │   ├── rcnn/
│   │   │   ├── model.py
│   │   ├── retinanet/
│   │   │   ├── model.py
│   ├── base_model.py
│   ├── utils.py
│
└── tests/              # Unit and integration tests


## 🛠 Setting Up the Development Environment

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/Curso-de-Robotica-e-IA/vespa.git
cd vespa
```

### 2️⃣ Install Dependencies
Using Poetry:
```bash
poetry install
```

### 3️⃣ Run Tests
```bash
pytest tests/
```