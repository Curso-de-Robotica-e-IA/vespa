from vespa.methods import detection
from vespa.datasets import detection_formats

class VespaAIModelManager:
    def __init__(self):
        self._task = None
        self._model = None
        self._dataset_format = None

    def set_task(self, task_type: str):
        if task_type not in ["detection"]:
            raise ValueError("Invalid task type. Choose from 'detection'.")
        self._task = task_type

    def set_dataset_format(self, dataset_format: str):
        if dataset_format not in ["yolo", "coco", "pascal_voc"]:
            raise ValueError("Invalid dataset format.")
        self._dataset_format = dataset_format

    def set_model(self, model_name: str):
        self._model = model_name

    def train(self, dataset_path: str):
        if self._task == "detection":
            dataset_kwargs = {"txt_file": "train.txt"}
            return detection.train_model(self._model, dataset_path, self._dataset_format, **dataset_kwargs)
        else:
            raise ValueError("No task type set. Use set_task().")
    
    def validate(self, dataset_path: str):
        if self._task == "detection":
            dataset_kwargs = {"txt_file": "val.txt"}
            return detection.validate_model(self._model, dataset_path, self._dataset_format, **dataset_kwargs)
        else:
            raise ValueError("No task type set. Use set_task().")

    def predict(self, image_path: str):
        if self._task == "detection":
            return detection.run_inference(self._model, image_path)
        else:
            raise ValueError("No task type set. Use set_task().")

    def list_models(self):
        return {"detection": detection.available_models()}

    def list_dataset_formats(self):
        return ["yolo", "coco", "pascal_voc"]
