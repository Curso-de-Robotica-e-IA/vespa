import math
import cv2 as cv
import numpy as np
import torch
import torchvision.transforms as transforms
from pathlib import Path

from PIL import Image
from skimage import feature

from vespa.methods.iqa.diqa.model.diqa_model import DIQAModel


class DIQA:
    def __init__(self, path: str):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.__device = torch.device(device)

        self.__model = DIQAModel()
        self.__attached_model = self.__model.to(self.__device)
        self.checkpoint_path = Path(path)

    def load(self, model_path: str) -> None:
        self.__model.load_state_dict(torch.load(model_path, weights_only=True))
        self.__attached_model = self.__model.to(self.__device)

    def predict(self, image_path: str) -> float:
        image = self.__load_images(image_path)
        prediction = None

        with torch.no_grad():
            target_image = image.to(self.__device)
            self.__attached_model.eval()
            output = self.__attached_model(target_image)
            prediction = output.item()

        prediction = 1 if prediction > 1 else prediction
        return prediction

    # internal function
    def __load_images(self, image_path: str):
        target_image = Image.open(image_path)

        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        target_image = transform(target_image)
        target_image = target_image.unsqueeze(0)

        return target_image

