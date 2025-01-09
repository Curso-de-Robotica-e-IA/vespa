from albumentations import (
    Affine,
    Compose,
    HorizontalFlip,
    Normalize,
    RandomBrightnessContrast,
)
from albumentations.pytorch import ToTensorV2

from vespa.datasets.config import (
    MEAN_PASCALVOC,
    STD_PASCALVOC,
    load_hyperparams,
)


def get_pascal_voc_train_transforms():
    """
    Transformações de treino específicas para o dataset Pascal VOC.
    """
    hyps = load_hyperparams('PascalVOC')

    return Compose(
        [
            HorizontalFlip(p=hyps.get('horizontal_flip', 0.6)),
            RandomBrightnessContrast(
                p=hyps.get('random_brightness_contrast', 0.4),
                brightness_limit=hyps.get('brightness_limit', 0.3),
                contrast_limit=hyps.get('contrast_limit', 0.3),
            ),
            Affine(
                p=hyps.get('shift_scale_rotate', 0.3),
                translate_percent={
                    'x': hyps.get('shift_limit', 0.05),
                    'y': hyps.get('shift_limit', 0.05),
                },
                scale=(
                    1 - hyps.get('scale_limit', 0.3),
                    1 + hyps.get('scale_limit', 0.3),
                ),
                rotate=(
                    -hyps.get('rotate_limit', 10),
                    hyps.get('rotate_limit', 10),
                ),
            ),
            Normalize(mean=MEAN_PASCALVOC, std=STD_PASCALVOC),
            ToTensorV2(),
        ],
        bbox_params={
            'format': 'pascal_voc',
            'label_fields': ['labels'],
            'min_visibility': hyps.get('min_visibility', 0.6),
        },
    )


def get_pascal_voc_test_transforms():
    """
    Transformações de teste/validação específicas para o dataset Pascal VOC.
    """
    return Compose([
        Normalize(mean=MEAN_PASCALVOC, std=STD_PASCALVOC),
        ToTensorV2(),
    ])