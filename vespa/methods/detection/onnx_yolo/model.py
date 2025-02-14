import onnxruntime as ort
import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple
from torch import Tensor, no_grad, load, save
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support

from vespa.datasets.base_dataset import BaseDataset
from vespa.methods.base_model import BaseModel
from vespa.methods.utils import configure_optimizer, custom_collate_fn


class YOLO(BaseModel):
    def __init__(
        self,
        model_path: str,
        classes: Dict[int, str],
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.5,
        *args,
        **kwargs,
    ):
        """
        Initializes the YOLO ONNX model.

        Args:
            model_path (str): Path to the ONNX model file.
            classes (Dict[int, str]): Dictionary mapping class IDs to class names.
            confidence_threshold (float): Minimum confidence score for detections.
            iou_threshold (float): Intersection over Union threshold for NMS.
        """
        super().__init__(*args, **kwargs)
        self.model_path = model_path
        self.classes = classes
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold

        # Load ONNX model
        self.session = ort.InferenceSession(self.model_path, providers=["CPUExecutionProvider"])

        # Get input/output names
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        # Retrieve model input size
        input_shape = self.session.get_inputs()[0].shape
        self.input_height, self.input_width = input_shape[2], input_shape[3]

        # Configure optimizer (Not used for ONNX, placeholder for compatibility)
        self.optimizer = None

    def forward(self, images: List[Tensor], targets: Optional[List[Dict[str, Tensor]]] = None) -> List[Dict[str, Tensor]]:
        """
        Runs inference on a batch of images.

        Args:
            images (List[Tensor]): List of input images as tensors.
            targets (Optional[List[Dict[str, Tensor]]]): Not used in inference.

        Returns:
            List[Dict[str, Tensor]]: List of detections per image.
        """
        return [self.detect(image.numpy()) for image in images]

    def fit(self, train_dataset: BaseDataset, batch_size: int, epochs: int = 20, device: str = "cuda") -> None:
        """
        Placeholder for training (ONNX models cannot be trained directly).

        Raises:
            NotImplementedError: Training is not supported for ONNX models.
        """
        raise NotImplementedError("YOLO ONNX does not support training. Use a PyTorch YOLO model for training.")

    @no_grad()
    def valid(self, val_dataset: BaseDataset, batch_size: int, device: str) -> float:
        """
        Validates the model on a validation dataset.

        Args:
            val_dataset (BaseDataset): Validation dataset.
            batch_size (int): Batch size.
            device (str): Device ('cuda' or 'cpu').

        Returns:
            float: Average validation loss.
        """
        raise NotImplementedError("YOLO ONNX does not support validation.")

    @no_grad()
    def test(self, test_dataset: BaseDataset, batch_size: int, device: str) -> Dict[str, float]:
        """
        Tests the YOLO ONNX model and computes evaluation metrics.

        Args:
            test_dataset (BaseDataset): Test dataset.
            batch_size (int): Batch size.
            device (str): Device ('cuda' or 'cpu').

        Returns:
            Dict[str, float]: Dictionary with precision, recall, and F1-score.
        """
        self.model.eval()
        self.model.to(device)

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=custom_collate_fn,
            num_workers=4,
            pin_memory=True,
        )

        all_preds = []
        all_labels = []

        for images, targets in test_loader:
            images = [img.to(device) for img in images]
            outputs = self.predict(images)

            for output, target in zip(outputs, targets):
                preds = [det["confidence"] for det in output]
                labels = target['labels'].cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels)

        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted'
        )

        metrics = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
        }

        print(f'Test Metrics: {metrics}')
        return metrics

    @no_grad()
    def predict(self, images: List[Tensor], device: str = 'cuda') -> List[Dict[str, Tensor]]:
        """
        Runs inference on a list of images.

        Args:
            images (List[Tensor]): List of input images.
            device (str): Device to use ('cuda' or 'cpu').

        Returns:
            List[Dict[str, Tensor]]: List of detection results.
        """
        images = [image.to(device) for image in images]
        results = []

        for image in images:
            input_tensor = image.unsqueeze(0).numpy()  # Convert to ONNX format
            outputs = self.session.run([self.output_name], {self.input_name: input_tensor})
            results.append(outputs)

        return results

    def save(self, path: str):
        """
        Saves the trained model (Not supported in ONNX).

        Raises:
            NotImplementedError: Saving an ONNX model from training is not supported.
        """
        raise NotImplementedError("Saving a trained model is not supported in ONNX runtime.")

    def load(self, path: str):
        """
        Loads a trained ONNX model.

        Args:
            path (str): Path to the ONNX model file.
        """
        self.session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])

    def print_model_summary(self):
        """
        Prints model information.
        """
        print("YOLO ONNX Model Path:", self.model_path)
        print("Number of Classes:", len(self.classes))
        print("Confidence Threshold:", self.confidence_threshold)
