import os
import numpy as np
import random
import torch
import torch.nn as nn
from typing import Tuple, List
from PIL import Image
from envs.ARNIQA.Lib.datetime import datetime
from torchvision import transforms
from tqdm import tqdm
from pathlib import Path

from vespa.methods.iqa.arniqa.model.resnet import ResNet
from vespa.methods.iqa.arniqa.model.simclr import SimCLR
from vespa.methods.iqa.iqa_model import IQABaseModel

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

SEED = 27


class ARNIQAModel(IQABaseModel):
    """
    ARNIQA model for No-Reference Image Quality Assessment (NR-IQA). It is composed of a ResNet-50 encoder and a Ridge
    regressor. The regressor is trained on the dataset specified by the parameter 'regressor_dataset'. The model takes
    in input an image both at full-scale and half-scale. The output is the predicted quality score. By default, the
    predicted quality scores are in the range [0, 1], where higher is better. In addition to the score, the forward
    function allows returning the concatenated embeddings of the image at full-scale and half-scale.
    """
    def __init__(self, model_weights_path: str, regressor_weights_path: str):
        super(ARNIQAModel, self).__init__()

        # Set seed
        torch.manual_seed(SEED)
        random.seed(SEED)
        torch.use_deterministic_algorithms(True)
        np.random.seed(SEED)
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"

        self.device = torch.device('cuda') if torch.cuda.is_available() else "cpu"
        self.encoder = ResNet(embedding_dim=128, use_norm=True)

        self.encoder.load_state_dict(torch.load(model_weights_path, map_location="cpu"))
        self.encoder.eval().to(self.device)

        self.regressor: nn.Module = torch.load(regressor_weights_path, map_location="cpu")
        self.regressor.eval().to(self.device)

        # Intialize the model
        self.model = SimCLR(self.encoder, temperature=0.1)
        self.model.to(self.device)

        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer,
                                                                                 T_0=1,
                                                                                 T_mult=2,
                                                                                 eta_min=1e-6,
                                                                                 verbose=False)
        self.scaler = torch.cuda.amp.GradScaler()

        self.checkpoint_path = "pretrain"

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

    def predict(self, image_path: str):
        img = Image.open(image_path).convert('RGB')

        # Get the halfof the image
        img_ds = transforms.Resize((img.size[1] // 2, img.size[0] // 2))(img)

        # Preprocess the images
        img = self.preprocess(img).unsqueeze(0).to(self.device)
        img_ds = self.preprocess(img_ds).unsqueeze(0).to(self.device)

        with torch.no_grad(), torch.amp.autocast("cuda"):
            score = self.model(img, img_ds, return_embedding=False, scale_score=True)
        return score.item()

    def train(self, train_dataset, batch_size, epochs, device):
        start_epoch = 0
        max_epochs = epochs
        best_srocc = 0
        last_srocc = 0
        last_plcc = 0
        last_model_filename = ""
        best_model_filename = ""

        # Training loop
        for epoch in range(start_epoch, max_epochs):
            self.model.train()
            running_loss = 0.0
            progress_bar = tqdm(train_dataset, desc=f"Epoch [{epoch +1}/{max_epochs}]")

            for i, batch in enumerate(tqdm(train_dataset)):
                num_logging_steps = i * batch_size + len(train_dataset) * batch_size * epoch

                # Initialize inputs
                inputs_A_orig = batch["img_A_orig"].to(device=self.device, non_blocking=True)
                inputs_A_ds = batch["img_A_ds"].to(device=self.device, non_blocking=True)
                inputs_A = torch.cat((inputs_A_orig, inputs_A_ds), dim=0)
                inputs_B_orig = batch["img_B_orig"].to(device=self.device, non_blocking=True)
                inputs_B_ds = batch["img_B_ds"].to(device=self.device, non_blocking=True)
                inputs_B = torch.cat((inputs_B_orig, inputs_B_ds), dim=0)
                img_A_name = batch["img_A_name"]
                img_B_name = batch["img_B_name"]

                distortion_functions = np.array(batch["distortion_functions"]).T  # Handle PyTorch's indexing of lists
                distortion_functions = [list(filter(None,el)) for el in distortion_functions]  # Remove padding
                distrotion_values = torch.stack(batch["distortion_values"]).T  # Handle PyTorch's indexing of lists
                distortion_values = [el[el != torch.inf] for el in distortion_values]  # Remove padding

                # Zero the parameter gradients
                self.optimizer.zero_grad()

                # Forward + backward + optimze
                with torch.amp.autocast("cuda"):
                    loss = self.model(inputs_A, inputs_B)

                if torch.isnan(loss):
                    raise ValueError("Loss is NaN")

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                if self.lr_scheduler and self.lr_scheduler.__class__.__name__ == "CosineAnnealingWarmRestarts":
                    self.lr_scheduler.step(int(epoch + i / len(train_dataset)))

                curr_loss = loss.item()
                running_loss += curr_loss
                progress_bar.set_postfix(loss=running_loss / (i + 1), SROCC=last_srocc, PLCC=last_plcc)

            if self.lr_scheduler and self.lr_scheduler.__class__.__name__ != "CosineAnnealingWarmRestarts":
                self.lr_scheduler.step()

            # Validation
            print("Starting validation...")
            last_srocc, last_plcc = self.valid()

            progress_bar.set_postfix(loss=running_loss / (i + 1), SROCC=last_srocc, PLCC=last_plcc)

            # Save checkpoints
            print("Saving checkpoint")

            # Save best checkpoint weights
            if last_srocc > best_srocc:
                best_srocc = last_srocc
                best_plcc = last_plcc
                if best_model_filename:
                    os.remove(self.checkpoint_path / best_model_filename)  # Remove previous best model
                    best_model_filename = f'best_epoch_{epoch}_srocc_{best_srocc:.3f}_plcc_{best_plcc:.3f}.pth'
                    torch.save(self.model.state_dict(), self.checkpoint_path / best_model_filename)

            # Save last checkpoint
            if last_model_filename:
                os.remove(self.checkpoint_path / last_model_filename)  # Remove previous last model
                last_model_filename = f"last_epoch_{epoch}_srocc_{last_srocc:.3f}_plcc_{last_plcc:.3f}.pth"
                torch.save({"model_state_dict": self.model.state_dict(),
                            "optimizer_state_dict": self.optimizer.state_dict(),
                           "scale_state_dict": self.scaler.state_dict(),
                            "epoch": epoch}, self.checkpoint_path / last_model_filename)

        print('Finished training')

    def valid(self, train_dataset, batch_size: int, device: str) -> Tuple[float, float]:
        self.model.eval()

        #srocc_all, plcc_all, _, _, _ = get_results()

    def get_results(self,
                    data_base_path: Path,
                    datasets: List[str],
                    num_splits: int,
                    phase: str,
                    alpha: float,
                    grid_search: bool,
                    crop_size: int,
                    batch_size: int,
                    num_workers: int,
                    device: torch.device,
                    eval_type: str = "scratch") -> Tuple[dict, dict, dict, dict, dict]:
        """
            Get the results for the given model and datasets. Depending on the phase parameter, can be used both for validation
            and test. If phase == 'test' and grid_search == True, performs a grid search over the validation splits to find the best
            alpha value for the regression for each dataset. The results related to synthetic datasets contain also the results
            for each distortion type.

            Args:
                data_base_path (pathlib.Path): base path of the datasets
                datasets (list): list of datasets
                num_splits (int): number of splits
                phase (str): phase of the datasets. Must be in ['val', 'test']
                alpha (float): alpha value to use for regression. During test, if None, performs a grid search
                grid_search (bool): whether to perform a grid search over the validation splits to find the best alpha value for the regression
                crop_size (int): crop size
                batch_size (int): batch size
                num_workers (int): number of workers for the dataloaders
                device (torch.device): device to use for testing
                eval_type (str): Whether to test a model trained from scratch or the one pretrained by the authors of the ARNIQA paper.

            Returns:
                srocc_all (dict): dictionary containing the SROCC results
                plcc_all (dict): dictionary containing the PLCC results
                regressors (dict): dictionary containing the regressors
                alphas (dict): dictionary containing the alpha values used for the regression
                best_worst_results_all (dict): dictionary containing the best and worst results
            """
        srocc_all = {}
        plcc_all = {}
        regressors = {}
        alphas = {}
        best_worst_results_all = {}

        assert phase in ["val", "test"], "Phase must be in ['val', 'test']"

        print(f"{datetime.now().strftime('%d/%m/%Y %H:%M:%S')} Starting {phase} phase")
        for d in datasets:
            if d == "live":
                dataset = "live"





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
