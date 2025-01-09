import yaml
from albumentations import Compose, HorizontalFlip, RandomBrightnessContrast, ShiftScaleRotate, Normalize
from albumentations.pytorch import ToTensorV2
from vespa.datasets.config import MEAN, STD, load_hyperparams


def get_pascal_voc_train_transforms():
    """
    Transformações de treino específicas para o dataset Pascal VOC.
    """
    hyps = load_hyperparams("PascalVOC")

    return Compose([
        HorizontalFlip(p=hyps.get("horizontal_flip", 0.6)),
        RandomBrightnessContrast(
            p=hyps.get("random_brightness_contrast", 0.4),
            brightness_limit=hyps.get("brightness_limit", 0.3),
            contrast_limit=hyps.get("contrast_limit", 0.3),
        ),
        ShiftScaleRotate(
            p=hyps.get("shift_scale_rotate", 0.4),
            shift_limit=hyps.get("shift_limit", 0.05),
            scale_limit=hyps.get("scale_limit", 0.3),
            rotate_limit=hyps.get("rotate_limit", 20),
        ),
        Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ], bbox_params={
        "format": "pascal_voc",
        "label_fields": ["labels"],
        "min_visibility": hyps.get("min_visibility", 0.6),
    })


def get_pascal_voc_test_transforms():
    """
    Transformações de teste/validação específicas para o dataset Pascal VOC.
    """
    return Compose([
        Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])
