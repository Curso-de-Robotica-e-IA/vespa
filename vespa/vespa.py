from vespa.methods import detection


class Vespa:
    def __init__(self):
        self._task = None
        self._model = None
        self._dataset_format = None

    def set_task(self, task_type: str):
        if not task_type == 'detection':
            raise ValueError("Invalid task type. Choose from 'detection'.")
        self._task = task_type

    def set_dataset_format(self, dataset_format: str):
        if dataset_format not in {'yolo', 'coco', 'pascal_voc'}:
            raise ValueError('Invalid dataset format.')
        self._dataset_format = dataset_format

    def set_model(self, model_name: str):
        self._model = model_name

    def train(self, dataset_path: str):
        if self._task == 'detection':
            return detection.train_model(
                self._model,
                dataset_path,
                self._dataset_format,
                txt_file='train.txt',
            )
        else:
            raise ValueError('No task type set. Use set_task().')

    def validate(self, dataset_path: str):
        if self._task == 'detection':
            return detection.validate_model(
                self._model,
                dataset_path,
                self._dataset_format,
                txt_file='val.txt',
            )
        else:
            raise ValueError('No task type set. Use set_task().')

    def predict(self, image_path: str):
        if self._task == 'detection':
            return detection.run_inference(self._model, image_path)
        else:
            raise ValueError('No task type set. Use set_task().')

    @staticmethod
    def list_models():
        return {'detection': detection.available_models()}

    @staticmethod
    def list_dataset_formats():
        return ['yolo', 'coco', 'pascal_voc']
