from typing import Dict, List, Optional

import torch
from torch.nn import Module
from torch.optim import SGD, Adagrad, Adam, RMSprop
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


class RCNN(Module):
    def __init__(
        self,
        num_classes: int = 9,
        weights: Optional[str] = 'DEFAULT',
        optimizer_name: str = 'adam',
        lr: float = 0.0001,
        weight_decay: float = 0.0001,
        **kwargs,
    ):
        """
        Initializes the RCNN model with customizable parameters.

        Args:
            num_classes (int): Number of target classes
                                (including background).
            weights (str, optional): Pre-trained weights to use.
                                     Defaults to "DEFAULT".
            optimizer_name (str): Name of the optimizer to use.
                                  Defaults to "adam".
            lr (float): Learning rate. Defaults to 0.0001.
            weight_decay (float): Weight decay (L2 penalty).
                                  Defaults to 0.0001.
            **kwargs: Additional keyword arguments for flexibility.
        """
        super().__init__()

        # Load pre-trained model
        self.model = fasterrcnn_resnet50_fpn_v2(weights=weights)

        # Replace the predictor head
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(
            in_features, num_classes
        )

        # Configure optimizer
        self.optimizer_name = optimizer_name
        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer = self.configure_optimizer()
        self.num_classes = num_classes

    def forward(
        self,
        images: List[torch.Tensor],
        targets: Optional[List[Dict[str, torch.Tensor]]] = None,
    ):  # noqa
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

    def train_model(  # noqa
        self,
        train_dataset,
        val_dataset=None,
        batch_size=4,
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
            collate_fn=lambda x: tuple(zip(*x)),
            num_workers=4,
            pin_memory=True,
        )

        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0

            for images, targets in train_loader:
                images = [img.to(device) for img in images]  # noqa
                targets = [
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

    def configure_optimizer(self):
        """
        Configure the optimizer based on the selected type.

        Returns:
            torch.optim.Optimizer: Configured optimizer instance.
        """
        optimizer_name = self.optimizer_name.lower()

        if optimizer_name == 'adam':
            return Adam(
                self.model.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
        elif optimizer_name == 'sgd':
            return SGD(
                self.model.parameters(),
                lr=self.lr,
                momentum=0.9,
                weight_decay=self.weight_decay,
            )
        elif optimizer_name == 'adagrad':
            return Adagrad(
                self.model.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
        elif optimizer_name == 'rmsprop':
            return RMSprop(
                self.model.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
        else:
            raise ValueError(f'Unsupported optimizer: {self.optimizer_name}')

    @torch.no_grad()
    def evaluate(self, val_dataset, batch_size=4, device='cuda'):
        """
        Evaluate the RCNN model.

        Args:
            val_dataset: Validation dataset.
            batch_size (int): Batch size. Defaults to 4.
            device (str): Device to evaluate on ("cuda" or "cpu").
                          Defaults to "cuda".

        Returns:
            List[Dict]: List of predictions.
        """
        self.model.eval()
        self.model.to(device)

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda x: tuple(zip(*x)),
            num_workers=4,
            pin_memory=True,
        )

        predictions = []
        for images, _ in val_loader:
            images = [img.to(device) for img in images]  # noqa
            outputs = self.model(images)
            predictions.extend(outputs)

        return predictions
