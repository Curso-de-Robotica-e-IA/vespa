import torch
import torch.nn as nn
from typing import Tuple

from vespa.methods.iqa.arniqa.model.resnet import ResNet


class ARNIQAPredictor(nn.Module):
    """
    ARNIQA model for No-Reference Image Quality Assessment (NR-IQA). It is composed of a ResNet-50 encoder and a Ridge
    regressor. The regressor is trained on the dataset specified by the parameter 'regressor_dataset'. The model takes
    in input an image both at full-scale and half-scale. The output is the predicted quality score. By default, the
    predicted quality scores are in the range [0, 1], where higher is better. In addition to the score, the forward
    function allows returning the concatenated embeddings of the image at full-scale and half-scale. Also, the model can
    return the unscaled score (i.e. in the range of the training dataset).
    """
    def __init__(self, model_weights_path, regressor_weights_path):
        super(ARNIQAPredictor, self).__init__()
        self.encoder = ResNet(embedding_dim=128, use_norm=True)

        self.encoder.load_state_dict(torch.load(model_weights_path, map_location="cpu"))
        self.encoder.eval()

        self.regressor: nn.Module = torch.load(regressor_weights_path, map_location="cpu")
        self.regressor.eval()

    def forward(self, img, img_ds, return_embedding: bool = False, scale_score: bool = True):
        f, _ = self.encoder(img)
        f_ds, _ = self.encoder(img_ds)
        f_combined = torch.hstack((f, f_ds))
        score = self.regressor(f_combined)
        if scale_score:
            score = self._scale_score(score)
        if return_embedding:
            return score, f_combined
        else:
            return score

    def _scale_score(self, score: float, new_range: Tuple[float, float] = (0., 1.)) -> float:
        """
        Scale the score in the range [0, 1], where higher is better.

        Args:
            score (float): score to scale
            new_range (Tuple[float, float]): new range of the scores
        """

        # Compute scaling factors
        original_range = (1, 100)
        original_width = original_range[1] - original_range[0]
        new_width = new_range[1] - new_range[0]
        scaling_factor = new_width / original_width

        # Scale score
        scaled_score = new_range[0] + (score - original_range[0]) * scaling_factor

        return scaled_score
