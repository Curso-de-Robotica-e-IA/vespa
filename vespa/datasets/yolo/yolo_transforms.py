import yaml
from albumentations import Compose, HorizontalFlip, RandomBrightnessContrast, Affine, Normalize
from albumentations.pytorch import ToTensorV2
from vespa.datasets.config import MEAN, STD, load_hyperparams


def get_yolo_train_transforms():
    """
    Transformações de treino específicas para o dataset YOLO.
    """
    hyps = load_hyperparams("YOLO")

    return Compose([
        HorizontalFlip(p=hyps.get("horizontal_flip", 0.5)),
        RandomBrightnessContrast(
            p=hyps.get("random_brightness_contrast", 0.4),
            brightness_limit=hyps.get("brightness_limit", 0.3),
            contrast_limit=hyps.get("contrast_limit", 0.3),
        ),
        Affine(
            p=hyps.get("shift_scale_rotate", 0.3),
            translate_percent={"x": hyps.get("shift_limit", 0.05), "y": hyps.get("shift_limit", 0.05)},
            scale=(1 - hyps.get("scale_limit", 0.2), 1 + hyps.get("scale_limit", 0.2)),
            rotate=(-hyps.get("rotate_limit", 15), hyps.get("rotate_limit", 15)),
        ),
        Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ], bbox_params={
        "format": "yolo",
        "label_fields": ["labels"],
        "min_visibility": hyps.get("min_visibility", 0.6),
    })


def get_yolo_test_transforms():
    """
    Transformações de teste/validação específicas para o dataset YOLO.
    """
    return Compose([
        Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])
