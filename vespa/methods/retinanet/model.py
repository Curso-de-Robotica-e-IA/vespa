from torchvision.models.detection import RetinaNet_ResNet50_FPN_V2_Weights, retinanet_resnet50_fpn_v2
from torchvision.models.resnet import ResNet50_Weights
from torch.nn import Module

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
            kwargs=kwargs
        )
    
    def forward(self, x):
        return self.model(x)
