## 📌 CLI Usage
Vespa provides a command-line interface for convenience.

### Checking Available Commands
```bash
vespa --help
```

### Running Inference via CLI
```bash
vespa predict --task classification --model resnet50 --input path/to/image.jpg
```

### Training a Model via CLI
```bash
vespa train --task detection --model yolo --dataset path/to/dataset
```
