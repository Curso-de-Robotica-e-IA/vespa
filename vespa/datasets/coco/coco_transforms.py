from albumentations import (
    Compose, HorizontalFlip, RandomBrightnessContrast, HueSaturationValue,
    Affine, Normalize
)
from albumentations.pytorch import ToTensorV2
from vespa.datasets.config import MEAN, STD, load_hyperparams


def get_coco_train_transforms():
    """
    Transformações de treino específicas para o dataset COCO.
    """
    hyps = load_hyperparams("COCO")

    return Compose(
        [
            HorizontalFlip(p=hyps.get("horizontal_flip", 0.4)),
            RandomBrightnessContrast(
                p=hyps.get("random_brightness_contrast", 0.3),
                brightness_limit=hyps.get("brightness_limit", 0.25),
                contrast_limit=hyps.get("contrast_limit", 0.25),
            ),
            Affine(
                p=hyps.get("shift_scale_rotate", 0.3),
                translate_percent={"x": hyps.get("shift_limit", 0.05), "y": hyps.get("shift_limit", 0.05)},
                scale=(1 - hyps.get("scale_limit", 0.2), 1 + hyps.get("scale_limit", 0.2)),
                rotate=(-hyps.get("rotate_limit", 10), hyps.get("rotate_limit", 10)),
            ),
            HueSaturationValue(
                p=hyps.get("hue_saturation_value", 0.4),
                hue_shift_limit=hyps.get("hue_shift_limit", 10),
                sat_shift_limit=hyps.get("sat_shift_limit", 15),
                val_shift_limit=hyps.get("val_shift_limit", 10),
            ),
            Normalize(mean=MEAN, std=STD),
            ToTensorV2(),
        ],
        bbox_params={
            "format": "pascal_voc",
            "label_fields": ["labels"],
            "min_visibility": hyps.get("min_visibility", 0.5),
        },
    )


def get_coco_test_transforms():
    """
    Transformações de teste/validação específicas para o dataset COCO.
    """
    return Compose([
        Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])
