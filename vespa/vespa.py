from vespa.methods import detection

_task = None
_model = None
_dataset_format = None

def set_task(task_type: str):
    global _task # noqa
    if not task_type == 'detection':
        raise ValueError("Invalid task type. Choose from 'detection'.")
    _task = task_type

def set_dataset_format(dataset_format: str):
    global _dataset_format # noqa
    if dataset_format not in {'yolo', 'coco', 'pascal_voc'}:
        raise ValueError('Invalid dataset format.')
    _dataset_format = dataset_format

def set_model(model_name: str):
    global _model # noqa
    _model = model_name

def train(dataset_path: str):
    if _task == 'detection':
        return detection.train_model(
            _model,
            dataset_path,
            _dataset_format,
            txt_file='train.txt',
        )
    else:
        raise ValueError('No task type set. Use set_task().')

def validate(dataset_path: str):
    if _task == 'detection':
        return detection.validate_model(
            _model,
            dataset_path,
            _dataset_format,
            txt_file='val.txt',
        )
    else:
        raise ValueError('No task type set. Use set_task().')

def predict(image_path: str):
    if _task == 'detection':
        return detection.run_inference(_model, image_path)
    else:
        raise ValueError('No task type set. Use set_task().')

def list_models():
    return {'detection': detection.available_models()}

def list_dataset_formats():
    return ['yolo', 'coco', 'pascal_voc']