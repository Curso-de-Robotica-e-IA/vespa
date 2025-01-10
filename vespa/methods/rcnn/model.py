from typing import List, Dict, Optional
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from vespa.methods.utils import configure_optimizer, custom_collate_fn
from vespa.methods.base_model import BaseModel


class RCNN(BaseModel):
    def __init__(
        self,
        num_classes: int = 9,
        weights: Optional[str] = 'DEFAULT',
        optimizer_name: str = 'adam',
        lr: float = 0.0001,
        weight_decay: float = 0.0001
        ):
        super().__init__()
        self.model = fasterrcnn_resnet50_fpn_v2(weights=weights)
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        self.optimizer = configure_optimizer(self.model, optimizer_name, lr, weight_decay)
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
            collate_fn=custom_collate_fn,
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
            collate_fn=custom_collate_fn,
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
            collate_fn=custom_collate_fn,
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

    def save(self, path: str):
        torch.save({'model_state_dict': self.model.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict()}, path)

    def load(self, path: str):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def print_model_summary(self):
        print(self.model)

    def count_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def freeze_backbone(self):
        """
        Freeze the backbone of the model to prevent updates during training.
        """
        for param in self.model.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """
        Unfreeze the backbone of the model to allow updates during training.
        """
        for param in self.model.backbone.parameters():
            param.requires_grad = True

    def adjust_learning_rate(self, new_lr: float):
        """
        Adjust the learning rate of the optimizer.

        Args:
            new_lr (float): New learning rate value.
        """
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
        print(f'Learning rate adjusted to {new_lr:.6f}')
