from torch.nn import Module
from torch.optim import LBFGS, SGD, Adadelta, Adagrad, Adam, RMSprop
from torch.utils.data import DataLoader
from torchvision.models.detection import (
    RetinaNet_ResNet50_FPN_V2_Weights,
    retinanet_resnet50_fpn_v2,
)
from torchvision.models.resnet import ResNet50_Weights


class RetinaNet(Module):
    def __init__(self, num_class=9, pre_trained=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if pre_trained:
            weights = RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1
            weights_backbone = ResNet50_Weights.IMAGENET1K_V2
        else:
            weights = None
            weights_backbone = None

        self.model = retinanet_resnet50_fpn_v2(
            weights=weights,
            weights_backbone=weights_backbone,
            num_class=num_class,
            args=args,
            kwargs=kwargs,
        )

        self.configure_optimzer()

    def forward(self, x):
        return self.model(x)

    def fit(
        self, train_dataset, batch_size, epochs=20, device=0
    ):
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

            # Calc and print avarage loss from epoch
            print(f'Average Loss: {epoch_loss / len(train_loader)}')

    def configure_optimzer(self, optim='adam', lr=0.0001, weight_decay=0.0001):
        """
        Configures the optimizer for the model.
        Parameters:
        optim (str): The type of optimizer to use. Options are 'adam', 'sgd',
        'adadelta', 'adagrad', 'lbfgs', and 'rmsprop'. Default is 'adam'.
        lr (float): The learning rate for the optimizer. Default is 0.0001.
        weight_decay (float): The weight decay (L2 penalty) for the optimizer.
        Default is 0.0001.
        Raises:
        ValueError: If an invalid optimizer type is provided.
        """
        if optim == 'adam':
            self.optimizer = Adam(
                self.model.parameters(), lr=lr, weight_decay=weight_decay
            )
        elif optim == 'sgd':
            self.optimizer = SGD(
                self.model.parameters(), lr=lr, weight_decay=weight_decay
            )
        elif optim == 'adadelta':
            self.optimizer = Adadelta(
                self.model.parameters(), lr=lr, weight_decay=weight_decay
            )
        elif optim == 'adagrad':
            self.optimizer = Adagrad(
                self.model.parameters(), lr=lr, weight_decay=weight_decay
            )
        elif optim == 'lbfgs':
            self.optimizer = LBFGS(
                self.model.parameters(), lr=lr, weight_decay=weight_decay
            )
        elif optim == 'rmsprop':
            self.optimizer = RMSprop(
                self.model.parameters(), lr=lr, weight_decay=weight_decay
            )
        else:
            raise ValueError('Invalid optimizer')
        )
    
    def forward(self, x):
        return self.model(x)
