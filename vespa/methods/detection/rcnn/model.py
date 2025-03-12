from typing import Dict, List, Optional

from sklearn.metrics import precision_recall_fscore_support
from torch import Tensor, load, no_grad, save
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2

from vespa.datasets.base_dataset import BaseDataset
from vespa.methods.base_model import BaseModel
from vespa.methods.utils import configure_optimizer, custom_collate_fn

NUM_CLASSES_COCO = 91

class RCNN(BaseModel):
    def __init__(
        self,
        num_classes: int = 9,
        weights: Optional[str] = 'DEFAULT',
        optimizer_name: str = 'adam',
        lr: float = 0.0001,
        weight_decay: float = 0.0001,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        # Define os pesos corretamente
        if num_classes != NUM_CLASSES_COCO:
            weights = None

        # Verifica se há pesos para o backbone nos argumentos
        weights_backbone = kwargs.get('weights_backbone', None)

        self.name = 'rcnn'

        self.model = fasterrcnn_resnet50_fpn_v2(
            weights=weights,
            weights_backbone=weights_backbone,
            num_classes=num_classes,
            args=args,
            kwargs=kwargs,
        )

        self.optimizer = configure_optimizer(
            self.model, optimizer_name, lr, weight_decay
        )

    def forward(
        self,
        images: List[Tensor],
        targets: Optional[List[Dict[str, Tensor]]] = None,
    ) -> Dict[str, Tensor]:
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

    def fit(
        self,
        train_dataset: BaseDataset,
        batch_size: int,
        epochs=20,
        device=0,
    ) -> None:
        """
        Train the model using the provided training dataset.
        Args:
            train_dataset (Dataset): The dataset to use for training.
            batch_size (int): The number of samples per batch to load.
            epochs (int, optional): The number of epochs to train the model.
            Default is 20.
            device (int or str, optional): The device to use for training (e.g.
            'cpu' or 'cuda:0'). Default is 0.
        Returns:
            None
        """

        # Load model on gpu
        self.model.to(device)

        # Create DataLoaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            collate_fn=lambda x: tuple(zip(*x)),
        )

        # Train loop
        for epoch in range(epochs):
            self.model.train()
            # Accumulate loss values for epochs
            epoch_loss = 0.0

            # Train dataloader loop
            for batch_idx, (images, targets) in enumerate(train_loader):
                # Create lists and pass images and ground truth to device
                images_list = list(image.to(device) for image in images)
                targets_list = [
                    {k: v.to(device) for k, v in t.items()} for t in targets
                ]

                # Calc loss train
                loss_dict = self.model(images_list, targets_list)
                losses = sum(loss for loss in loss_dict.values())

                # Backpropagation
                self.optimizer.zero_grad()
                losses.backward()
                self.optimizer.step()

                # Sum loss to accumulate
                epoch_loss += losses.item()

                # Show loss batch informations
                # Keep the batch train prints on same bash line
                print('\033[2K\r', end='', flush=True)
                print(
                    f'Epoch [{epoch + 1}/{epochs}], Batch [{batch_idx + 1}/{len(train_loader)}] Loss: {losses.item()}\033[0m',  # noqa
                    end=' ',
                    flush=True,
                )

            # Calc and print average loss from epoch
            print(f'Average Loss: {epoch_loss / len(train_loader)}')

    @no_grad()
    def valid(self, val_dataset, batch_size, device) -> float:
        self.model.eval()
        self.model.to(device)

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            collate_fn=lambda x: tuple(zip(*x)),
        )

        val_loss = 0.0
        for images, targets in val_loader:
            image_list = [img.to(device) for img in images]
            target_list = [
                {k: v.to(device) for k, v in t.items()} for t in targets
            ]

            loss_dict = self.model(image_list, target_list)
            losses = sum(loss for loss in loss_dict.values())

            val_loss += losses.item()

        return val_loss / len(val_loader)

    @no_grad()
    def test(
        self, test_dataset, batch_size: int, device: str
    ) -> Dict[str, float]:
        """
        Test the RCNN model and compute evaluation metrics.

        Args:
            test_dataset: Test dataset.
            batch_size (int): Batch size. Defaults to 4.
            device (str): Device to test on ('cuda' or 'cpu').

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
            image_list = [img.to(device) for img in images]
            outputs = self.model(image_list)

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

    @no_grad()
    def predict(
        self, images: List[Tensor], device: str = 'cuda'
    ) -> List[Dict[str, Tensor]]:
        images = [image.to(device) for image in images]
        self.model.to(device)

        self.model.eval()
        return self.model(images)

    def save(self, path: str):
        save(self.model.state_dict(), path)

    def load(self, path: str):
        checkpoint = load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def print_model_summary(self):
        print(self.model)
