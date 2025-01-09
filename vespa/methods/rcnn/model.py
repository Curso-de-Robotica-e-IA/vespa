from torch.nn import Module
from torch.optim import LBFGS, SGD, Adadelta, Adagrad, Adam, RMSprop
from torch.utils.data import DataLoader
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn
from torchvision.models.resnet import ResNet50_Weights


class RCNN(Module):
    def __init__(self, num_classes=9, weights="COCO_V1", *args, **kwargs):
        """
        Inicializa a classe RCNN com o modelo Faster R-CNN ResNet50 FPN.

        Args:
            num_classes (int): Número de classes para o modelo.
            weights (str): Pesos pré-treinados. Use "COCO_V1".
            *args: Argumentos adicionais.
            **kwargs: Argumentos adicionais de palavra-chave.
        """
        super().__init__(*args, **kwargs)

        if weights == "COCO_V1":
            weights = FasterRCNN_ResNet50_FPN_Weights.COCO_V1
            weights_backbone = ResNet50_Weights.IMAGENET1K_V1
        else:
            weights = None
            weights_backbone = None

        self.model = fasterrcnn_resnet50_fpn(
            weights=weights,
            weights_backbone=weights_backbone,
            num_classes=num_classes,
            args=args,
            kwargs=kwargs,
        )

        self.configure_optimizer()

    def forward(self, x):
        """
        Realiza a passagem forward no modelo.

        Args:
            x: Entrada do modelo.

        Returns:
            Saída do modelo.
        """
        return self.model(x)

    def train(self, train_dataset, val_dataset=None, batch_size=16, epochs=20, device=0):
        """
        Treina o modelo usando o conjunto de dados de treino.

        Args:
            train_dataset: Dataset de treinamento.
            val_dataset: Dataset de validação.
            batch_size (int): Tamanho do batch.
            epochs (int): Número de épocas.
            device: Dispositivo para treinamento (CPU ou GPU).
        """
        self.model.to(device)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            collate_fn=lambda x: tuple(zip(*x)),
        )

        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0

            for batch_idx, (images, targets) in enumerate(train_loader):
                images_list = [image.to(device) for image in images]
                targets_list = [
                    {k: v.to(device) for k, v in t.items()} for t in targets
                ]

                loss_dict = self.model(images_list, targets_list)
                losses = sum(loss for loss in loss_dict.values())

                self.optimizer.zero_grad()
                losses.backward()
                self.optimizer.step()

                epoch_loss += losses.item()

                print("\033[2K\r", end="", flush=True)
                print(
                    f"Epoch [{epoch + 1}/{epochs}], Batch [{batch_idx + 1}/{len(train_loader)}] Loss: {losses.item()}\033[0m",
                    end=" ",
                    flush=True,
                )

            print(f"Average Loss: {epoch_loss / len(train_loader)}")

    def configure_optimizer(self, optim="adam", lr=0.0001, weight_decay=0.0001):
        """
        Configura o otimizador para o modelo.

        Args:
            optim (str): Tipo de otimizador. Opções: 'adam', 'sgd', 'adadelta', 'adagrad', 'lbfgs', 'rmsprop'.
            lr (float): Taxa de aprendizado.
            weight_decay (float): Penalidade L2.

        Raises:
            ValueError: Se um tipo de otimizador inválido for fornecido.
        """
        if optim == "adam":
            self.optimizer = Adam(
                self.model.parameters(), lr=lr, weight_decay=weight_decay
            )
        elif optim == "sgd":
            self.optimizer = SGD(
                self.model.parameters(), lr=lr, weight_decay=weight_decay
            )
        elif optim == "adadelta":
            self.optimizer = Adadelta(
                self.model.parameters(), lr=lr, weight_decay=weight_decay
            )
        elif optim == "adagrad":
            self.optimizer = Adagrad(
                self.model.parameters(), lr=lr, weight_decay=weight_decay
            )
        elif optim == "lbfgs":
            self.optimizer = LBFGS(
                self.model.parameters(), lr=lr, weight_decay=weight_decay
            )
        elif optim == "rmsprop":
            self.optimizer = RMSprop(
                self.model.parameters(), lr=lr, weight_decay=weight_decay
            )
        else:
            raise ValueError("Invalid optimizer")
