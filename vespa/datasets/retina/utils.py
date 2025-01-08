import numpy as np
import cv2
import yaml

from torchvision import transforms
from torch import Tensor
from albumentations import Compose, HorizontalFlip, RandomBrightnessContrast, Affine, Normalize, HueSaturationValue
from albumentations.pytorch import ToTensorV2

# Parâmetros usados na normalização
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

def get_train_transforms(hyperparameters_yaml_path='hyps.yaml'):
    with open(hyperparameters_yaml_path, 'r') as file:
        hyps = yaml.safe_load(file)

    return Compose([
        HorizontalFlip(p=hyps['horizontal_flip']),
        RandomBrightnessContrast(p=hyps['random_brightness_contrast'], 
                                 brightness_limit=hyps['brightness_limit'], 
                                 contrast_limit=hyps['contrast_limit']
                                 ),
        HueSaturationValue(p=hyps['hue_saturation_value'],
                           hue_shift_limit=hyps['hue_shift_limit'],
                           sat_shift_limit=hyps['sat_shift_limit'],
                           val_shift_limit=hyps['val_shift_limit'],
                           ),
        Affine(
            p=hyps['shift_scale_rotate'],
            translate_percent=hyps['shift_limit'],
            scale=hyps['scale_limit'],
            rotate=hyps['rotate_limit'],
        ),
        Normalize(mean=mean.tolist(), std=std.tolist()),
        ToTensorV2(),
    ], bbox_params={
        'format': 'pascal_voc',  # Formato das caixas: [xmin, ymin, xmax, ymax]
        'label_fields': ['labels'],
        'min_visibility':hyps['min_visibility'],  
    })

def get_val_transforms():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean.tolist(), std=std.tolist()),
    ])

def show_transforms(img, boxes):
    # Converte tensor ou imagem normalizada para formato visualizável
    if isinstance(img, Tensor):
        img = img.permute(1, 2, 0).cpu().numpy()  # De [C, H, W] para [H, W, C]

    # Reverte a normalização
    img = (img * std + mean)  # Reescala para valores entre [0, 1]
    img = np.clip(img * 255, 0, 255).astype(np.uint8)  # Converte para [0, 255] e garante valores válidos
    
    # Converte de RGB para BGR (compatível com OpenCV)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Desenha as caixas delimitadoras
    for box in boxes:
        xmin, ymin, xmax, ymax = map(int, box)
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)  # Caixa verde

    # Exibe a imagem
    cv2.imshow('Transformed Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()