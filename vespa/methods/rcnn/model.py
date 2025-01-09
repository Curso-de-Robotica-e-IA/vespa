from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.resnet import ResNet50_Weights
from torch.nn import Module


class RCNN(Module):
    def __init__(self, num_classes=9, weights="DEFAULT", *args, **kwargs):
        """
        Initializes the RCNN model with a ResNet-50 backbone.

        Args:
            num_classes (int): Number of output classes (including background).
            weights (str or None): Pretrained weights to use. Options are:
                - "DEFAULT": Use the latest pretrained weights.
                - None: No pretrained weights.
        """
        super().__init__(*args, **kwargs)
        
        if weights == "DEFAULT":
            model_weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
            backbone_weights = ResNet50_Weights.DEFAULT
        else:
            model_weights = None
            backbone_weights = None

        self.model = fasterrcnn_resnet50_fpn_v2(
            weights=model_weights,
            weights_backbone=backbone_weights,
            num_classes=num_classes,
            **kwargs
        )

    def forward(self, x):
        """
        Defines the forward pass of the model.

        Args:
            x (Tensor): Input tensor (image batch).

        Returns:
            dict or list: Predictions or losses based on the mode.
        """
        return self.model(x)
