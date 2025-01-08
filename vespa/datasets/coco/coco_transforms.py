import yaml
from albumentations import Compose, HorizontalFlip, RandomBrightnessContrast, HueSaturationValue, ShiftScaleRotate, Normalize
from albumentations.pytorch import ToTensorV2
from vespa.datasets.config import MEAN, STD

def get_coco_train_transforms(hyps_path="./vespa/datasets/coco/hyps_coco.yaml"):
    """
    Transformações para treinamento no formato COCO.
    """
    with open(hyps_path, "r") as f:
        hyps = yaml.safe_load(f)

    return Compose([
        HorizontalFlip(p=hyps['horizontal_flip']),
        RandomBrightnessContrast(
            p=hyps['random_brightness_contrast'],
            brightness_limit=hyps['brightness_limit'],
            contrast_limit=hyps['contrast_limit'],
        ),
        HueSaturationValue(
            p=hyps['hue_saturation_value'],
            hue_shift_limit=hyps['hue_shift_limit'],
            sat_shift_limit=hyps['sat_shift_limit'],
            val_shift_limit=hyps['val_shift_limit'],
        ),
        ShiftScaleRotate(
            p=hyps['shift_scale_rotate'],
            shift_limit=hyps['shift_limit'],
            scale_limit=hyps['scale_limit'],
            rotate_limit=hyps['rotate_limit'],
        ),
        Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ], bbox_params={
        'format': 'coco',
        'label_fields': ['labels'],
        'min_visibility': hyps['min_visibility'],
    })

def get_coco_val_test_transforms():
    """
    Transformações para validação e teste no formato COCO.
    """
    return Compose([
        Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ], bbox_params={
        'format': 'coco',
        'label_fields': ['labels'],
    })
