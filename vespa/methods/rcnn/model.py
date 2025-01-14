from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from vespa.methods.base_model import BaseModel


class RCNN(BaseModel):
    def __init__(
        self,
        num_classes: int = 9,
        weights: Optional[str] = 'DEFAULT',
        optimizer_name: str = 'adam',
        lr: float = 0.0001,
        weight_decay: float = 0.0001,
    ):
        """
        Initialize the RCNN model with the given parameters.

        Args:
            num_classes (int): Number of classes for detection.
                            Defaults to 9.
            weights (Optional[str]): Pretrained weights to use.
                            Defaults to 'DEFAULT'.
            optimizer_name (str): Name of the optimizer.
                            Defaults to 'adam'.
            lr (float): Learning rate for the optimizer.
                            Defaults to 0.0001.
            weight_decay (float): Weight decay for the optimizer.
                            Defaults to 0.0001.
        """
        super().__init__()
        self.model = fasterrcnn_resnet50_fpn_v2(weights=weights)
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(
            in_features, num_classes
        )
        self.optimizer = self.configure_optimizer(
            self.model, optimizer_name, lr, weight_decay
        )
        self.num_classes = num_classes

    def forward(
        self,
        images: List[torch.Tensor],
        targets: Optional[List[Dict[str, torch.Tensor]]] = None,
    ):
        """
        Forward pass for the RCNN model.

        Args:
            images (List[torch.Tensor]):
                List of input images as tensors.
            targets (List[Dict[str, torch.Tensor]], optional):
                Target annotations for training.

        Returns:
            If training, returns a dict of losses. Otherwise,
            returns detections.
        """
        return self.model(images, targets)

    def train(
        self,
        train_dataset,
        batch_size=0,
        epochs=10,
        device: str = 'cuda',
        grad_clip: Optional[float] = None,
    ):
        """
        Train the RCNN model.

        Args:
            train_dataset: Training dataset.
            val_dataset: Validation dataset (optional).
            batch_size (int): Batch size. Defaults to 4.
            epochs (int): Number of epochs. Defaults to 10.
            device (str): Device to train on ("cuda" or "cpu").
                            Defaults to "cuda".
            grad_clip (float, optional): Max gradient norm for
                            gradient clipping. Defaults to None.
        """
        self.model.to(device)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=self.custom_collate_fn,
            num_workers=4,
            pin_memory=True,
        )

        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0

            for images, targets in train_loader:
                images = [img.to(device) for img in images]  # noqa
                targets = [  # noqa
                    {k: v.to(device) for k, v in t.items()} for t in targets
                ]  # noqa

                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

                self.optimizer.zero_grad()
                losses.backward()

                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), grad_clip
                    )

                self.optimizer.step()

                epoch_loss += losses.item()

            avg_loss = epoch_loss / len(train_loader)
            print(
                f'Epoch [{epoch + 1}/{epochs}] - Average Loss: {avg_loss:.4f}'
            )

    def valid(self, val_dataset, batch_size: int, device: str):
        """
        Validate the model and compute average loss on the validation set.

        Args:
            val_dataset: Validation dataset.
            batch_size (int): Batch size. Defaults to 4.
            device (str): Device to evaluate on ('cuda' or 'cpu').
                          Defaults to 'cuda'.

        Returns:
            float: Average validation loss.
        """
        self.model.eval()
        self.model.to(device)

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self.custom_collate_fn,
            num_workers=4,
            pin_memory=True,
        )

        val_loss = 0.0
        with torch.no_grad():
            for images, targets in val_loader:
                images = [img.to(device) for img in images]  # noqa
                targets = [  # noqa
                    {k: v.to(device) for k, v in t.items()}
                    for t in targets  # noqa
                ]

                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                val_loss += losses.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f'Validation Loss: {avg_val_loss:.4f}')
        return avg_val_loss

    def test(self, test_dataset, batch_size: int, device: str):
        """
        Test the RCNN model and compute evaluation metrics.

        Args:
            test_dataset: Test dataset.
            batch_size (int): Batch size. Defaults to 4.
            device (str): Device to test on ('cuda' or 'cpu').
                          Defaults to 'cuda'.

        Returns:
            Dict[str, float]: Dictionary containing evaluation metrics.
        """

        self.model.eval()
        self.model.to(device)

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self.custom_collate_fn,
            num_workers=4,
            pin_memory=True,
        )

        all_preds = []
        all_labels = []

        for images, targets in test_loader:
            images = [img.to(device) for img in images]  # noqa
            outputs = self.model(images)

            for output, target in zip(outputs, targets):
                preds = output['labels'].cpu().numpy()
                labels = target['labels'].cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels)

        metrics = self.calculate_metrics(all_labels, all_preds)

        print(f'Test Metrics: {metrics}')
        return metrics

    @torch.no_grad()
    def predict(
        self, images: List[torch.Tensor], device: str = 'cuda'
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Performs inferences on the model given an unlabeled dataset.

        Args:

            images (List[torch.Tensor]): List of image tensors.
            device (str): Device for inference ('cuda' or 'cpu').

        Returns:

            List[Dict[str, torch.Tensor]]: List of predictions for each image.
        """
        self.model.eval()
        self.model.to(device)

        images = [img.to(device) for img in images]
        outputs = self.model(images)

        return outputs
