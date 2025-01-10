from typing import Dict, List, Optional

import torch
from sklearn.metrics import precision_recall_fscore_support
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
            collate_fn=self.custom_collate_fn,
            num_workers=4,
            pin_memory=True,
        )

        predictions = []
        for images, _ in val_loader:
            images = [img.to(device) for img in images]  # noqa
            outputs = self.model(images)
            predictions.extend(outputs)

        return predictions

    def validate_model(self, val_dataset, batch_size=4, device='cuda'):
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

    def save_model(self, path: str):
        """
        Save the model and optimizer state.

        Args:
            path (str): Path to save the model.
        """
        torch.save(
            {
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
            },
            path,
        )

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

    def print_model_summary(self):
        """
        Print a summary of the model structure.
        """
        print(self.model)

    def count_trainable_parameters(self):
        """
        Count the number of trainable parameters in the model.

        Returns:
            int: Number of trainable parameters.
        """
        return sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )  # noqa

    def set_optimizer(
        self,
        optimizer_name: str,
        lr: float = 0.0001,
        weight_decay: float = 0.0001,
    ):
        """
        Set a new optimizer for the model.

        Args:
            optimizer_name (str): Name of the optimizer.
            lr (float): Learning rate. Defaults to 0.0001.
            weight_decay (float): Weight decay (L2 penalty).
                                  Defaults to 0.0001.
        """
        self.optimizer_name = optimizer_name
        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer = self.configure_optimizer()
        print(f'Optimizer set to {optimizer_name} with learning rate {lr:.6f}')
        
    def custom_collate_fn(self, batch):
        """
        Função de colagem para combinar imagens e anotações.
        """
        return tuple(zip(*batch))

    @torch.no_grad()
    def test_model(self, test_dataset, batch_size=4, device='cuda'):
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
