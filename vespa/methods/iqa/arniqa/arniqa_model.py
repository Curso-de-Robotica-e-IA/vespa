import os
import pickle

import numpy as np
import random
import torch
from typing import Tuple, List
from PIL import Image
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
from pathlib import Path
from sklearn.linear_model import Ridge
from einops import rearrange
from scipy import stats

from vespa.methods.iqa.arniqa.model.resnet import ResNet
from vespa.methods.iqa.arniqa.model.simclr import SimCLR
from vespa.methods.iqa.arniqa.model.arniqa_predictor import ARNIQAPredictor
from vespa.methods.iqa.iqa_model import IQABaseModel
from vespa.datasets.iqa_datasets import (LIVEDataset, CSIQDataset, TID2013Dataset, KADID10KDataset, FLIVEDataset,
                                         SPAQDataset, Koniq10kDataset, KADIS700Dataset)

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

SEED = 27
DATA_BASE_PATH = Path(r'\\192.168.155.240\Robotica\dataset_iqa')
NUM_SPLITS = 10
ALPHA = 0.1
VAL_DATASETS = ['live']

synthetic_datasets = ["live", "csiq", "tid2013", "kadid10k"]
authentic_datasets = ["flive", "spaq", "koniq10k"]


class ARNIQAModel(IQABaseModel):
    """
    ARNIQA model for No-Reference Image Quality Assessment (NR-IQA). It is composed of a ResNet-50 encoder and a Ridge
    regressor. The regressor is trained on the dataset specified by the parameter 'regressor_dataset'. The model takes
    in input an image both at full-scale and half-scale. The output is the predicted quality score. By default, the
    predicted quality scores are in the range [0, 1], where higher is better. In addition to the score, the forward
    function allows returning the concatenated embeddings of the image at full-scale and half-scale.
    """
    def __init__(self, checkpoint_path: str):
        super(ARNIQAModel, self).__init__()

        # Set seed
        torch.manual_seed(SEED)
        random.seed(SEED)
        torch.use_deterministic_algorithms(True)
        np.random.seed(SEED)
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"

        self.device = torch.device('cuda') if torch.cuda.is_available() else "cpu"
        self.encoder = ResNet(embedding_dim=128, use_norm=True)
        self.regressor = None
        self.arniqa_predictor = None
        # Intialize the model
        self.clr = SimCLR(self.encoder, temperature=0.1)
        self.clr.to(self.device)

        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.optimizer = torch.optim.SGD(self.clr.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer,
                                                                                 T_0=1,
                                                                                 T_mult=2,
                                                                                 eta_min=1e-6,
                                                                                 verbose=False)
        self.scaler = torch.cuda.amp.GradScaler()

        self.checkpoint_path = Path(checkpoint_path)
        self.train_dataloader = None
        self.weights_path = None

    def _load_kadis700(self):
        kadis_dataset = KADIS700Dataset(root=f'{DATA_BASE_PATH}/KADIS700',
                                             patch_size=224,
                                             max_distortions=4,
                                             num_levels=5,
                                             pristine_prob=0.05)
        train_dataloader = DataLoader(kadis_dataset, batch_size=4, num_workers=4, shuffle=True,
                                      pin_memory=True, drop_last=True)
        self.train_dataloader = train_dataloader

    def load(self, model_path: str, regressor_path: str):
        self.arniqa_predictor = ARNIQAPredictor(model_path, regressor_path)
        self.arniqa_predictor.eval().to(self.device)
        self.weights_path = model_path

    def predict(self, image_path: str):
        img = Image.open(image_path).convert('RGB')

        # Get the half of the image
        img_ds = transforms.Resize((img.size[1] // 2, img.size[0] // 2))(img)

        # Preprocess the images
        img = self.preprocess(img).unsqueeze(0).to(self.device)
        img_ds = self.preprocess(img_ds).unsqueeze(0).to(self.device)

        with torch.no_grad(), torch.amp.autocast("cuda"):
            self.arniqa_predictor.eval()
            score = self.arniqa_predictor(img, img_ds, return_embedding=False, scale_score=True)
        return score.item()

    def train(self, batch_size, epochs):
        start_epoch = 0
        max_epochs = epochs
        best_srocc = 0
        last_srocc = 0
        last_plcc = 0
        last_model_filename = ""
        best_model_filename = ""


        self._load_kadis700()

        # Training loop
        for epoch in range(start_epoch, max_epochs):
            self.clr.train()
            running_loss = 0.0
            progress_bar = tqdm(self.train_dataloader, desc=f"Epoch [{epoch +1}/{max_epochs}]")

            for i, batch in enumerate(tqdm(self.train_dataloader)):
                num_logging_steps = i * batch_size + len(self.train_dataloader) * batch_size * epoch

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
                distortion_values = torch.stack(batch["distortion_values"]).T  # Handle PyTorch's indexing of lists
                distortion_values = [el[el != torch.inf] for el in distortion_values]  # Remove padding

                # Zero the parameter gradients
                self.optimizer.zero_grad()

                # Forward + backward + optimze
                with torch.amp.autocast("cuda"):
                    loss = self.clr(inputs_A, inputs_B)

                if torch.isnan(loss):
                    raise ValueError("Loss is NaN")

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                if self.lr_scheduler and self.lr_scheduler.__class__.__name__ == "CosineAnnealingWarmRestarts":
                    self.lr_scheduler.step(int(epoch + i / len(self.train_dataloader)))

                curr_loss = loss.item()
                running_loss += curr_loss
                progress_bar.set_postfix(loss=running_loss / (i + 1), SROCC=last_srocc, PLCC=last_plcc)

            if self.lr_scheduler and self.lr_scheduler.__class__.__name__ != "CosineAnnealingWarmRestarts":
                self.lr_scheduler.step()

            # Validation
            print("Starting validation...")
            last_srocc, last_plcc = self.valid(batch_size)

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
                    torch.save(self.clr.state_dict(), self.checkpoint_path / best_model_filename)
                    self.weights_path = self.checkpoint_path / best_model_filename

            # Save last checkpoint
            if last_model_filename:
                os.remove(self.checkpoint_path / last_model_filename)  # Remove previous last model
                last_model_filename = f"last_epoch_{epoch}_srocc_{last_srocc:.3f}_plcc_{last_plcc:.3f}.pth"
                torch.save({"model_state_dict": self.clr.state_dict(),
                            "optimizer_state_dict": self.optimizer.state_dict(),
                           "scale_state_dict": self.scaler.state_dict(),
                            "epoch": epoch}, self.checkpoint_path / last_model_filename)

        print('Finished training')

    def valid(self, batch_size: int) -> Tuple[float, float]:
        """
        Validate the given model on the validation datasets.

        Args:
            batch_size (int): Batch size used for validation.
        """
        self.clr.eval()

        srocc_all, plcc_all, _, _, _ = self.get_results(data_base_path=DATA_BASE_PATH, datasets=VAL_DATASETS,
                                                        num_splits=NUM_SPLITS, phase="val", alpha=ALPHA,
                                                        grid_search=False, crop_size=224, batch_size=batch_size,
                                                        num_workers=0)

        # Compute the median for each list in srocc_all and plcc_all
        srocc_all_median = {key: np.median(value["global"]) for key, value in srocc_all.items()}
        plcc_all_median = {key: np.median(value['global']) for key, value in plcc_all.items()}

        # Compute the global average
        srocc_avg = np.mean(list(srocc_all_median.values()))
        plcc_avg = np.mean(list(plcc_all_median.values()))

        return srocc_avg, plcc_avg

    def test(self, batch_size: int):
        """
            Test pretrained model on the test datasets. Performs a grid search over the validation splits to find the best
            alpha value for the regression for each dataset.

            Args:
                batch_size (int): Batch size used for training.
        """

        if self.weights_path:
            checkpoint = torch.load(self.weights_path)
            self.clr.load_state_dict(checkpoint, strict=True)

        self.clr.eval()
        self.clr.to(self.device)

        sroc_all, plcc_all, regressors, alphas, best_worst_results_all = self.get_results(data_base_path=DATA_BASE_PATH,
                                                                                          datasets=VAL_DATASETS,
                                                                                          num_splits=NUM_SPLITS,
                                                                                          phase="test",
                                                                                          alpha=ALPHA,
                                                                                          grid_search=True,
                                                                                          crop_size=224,
                                                                                          batch_size=batch_size,
                                                                                          num_workers=0,
                                                                                          eval_type="scratch")

        # Compute the median for each list in srocc_all and plcc_all
        srocc_all_median = {key: np.median(value["global"]) for key, value in sroc_all.items()}
        plcc_all_median = {key: np.median(value["global"]) for key, value in plcc_all.items()}

        # Compute the synthetic and autentic averages
        srocc_synthetic_avg = np.mean(
            [srocc_all_median[key] for key in srocc_all_median.keys() if key in synthetic_datasets])
        plcc_synthetic_avg = np.mean(
            [plcc_all_median[key] for key in plcc_all_median.keys() if key in synthetic_datasets])
        srocc_authentic_avg = np.mean(
            [srocc_all_median[key] for key in srocc_all_median.keys() if key in authentic_datasets])
        plcc_authentic_avg = np.mean(
            [plcc_all_median[key] for key in plcc_all_median.keys() if key in authentic_datasets])

        # Compute the global average
        srocc_avg = np.mean(list(srocc_all_median.values()))
        plcc_avg = np.mean(list(plcc_all_median.values()))

        print(f"{'Dataset':<15} {'Alpha':<15} {'SROCC':<15} {'PLCC':<15}")
        for dataset in srocc_all_median.keys():
            print(f"{dataset:<15} {alphas[dataset]} {srocc_all_median[dataset]:<15.4f} {plcc_all_median[dataset]:<15.4f}")
        print(f"{'Synthetic avg':<15} {srocc_synthetic_avg:<15.4f} {plcc_synthetic_avg:<15.4f}")
        print(f"{'Authentic avg':<15} {srocc_authentic_avg:<15.4f} {plcc_authentic_avg:<15.4f}")

        for dataset, regressor in regressors.items():
            filename = (f"{datetime.now().strftime('%d_%m_%Y')}_{dataset}_srocc_"
                        f"{srocc_all_median[dataset]:.4f}_plcc_{plcc_all_median[dataset]:.4f}.pkl")
            with open(str(self.checkpoint_path / filename), "wb") as f:
                pickle.dump(regressor, f)

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
                dataset = LIVEDataset(data_base_path / "LIVE", phase="all", crop_size=crop_size)
                dataset_num_splits = num_splits
                dataset_name = "LIVE"
            elif d == "csiq":
                dataset = CSIQDataset(data_base_path / "CSIQ", phase="all", crop_size=crop_size)
                dataset_num_splits = num_splits
                dataset_name = "CSIQ"
            elif d == "tid2013":
                dataset = TID2013Dataset(data_base_path / "TID2013", phase="all", crop_size=crop_size)
                dataset_num_splits = num_splits
                dataset_name = "TID2013"
            elif d == "kadid10k":
                dataset = KADID10KDataset(data_base_path / "KADID10K", phase="all", crop_size=crop_size)
                dataset_num_splits = num_splits
                dataset_name = "KADID-10K"
            elif d == "flive":
                dataset = FLIVEDataset(data_base_path / "FLIVE", phase="all", crop_size=crop_size)
                dataset_num_splits = 1
                dataset_name = "FLIVE"
            elif d == 'spaq':
                dataset = SPAQDataset(data_base_path / "spaq", phase="all", crop_size=crop_size)
                dataset_num_splits = num_splits
                dataset_name = "SPAQ"
            elif d == 'koniq10k':
                dataset = Koniq10kDataset(data_base_path / "KonIQ-10k", phase="all", crop_size=crop_size)
                dataset_num_splits = num_splits
                dataset_name = "KONIQ10K"
            else:
                raise ValueError(f"Dataset {d} not recognized")

            srocc_dataset, plcc_dataset, regressor, alpha, best_worst_results = self.compute_metrics(dataset,
                                                                                                     dataset_num_splits,
                                                                                                     phase, alpha,
                                                                                                     grid_search,
                                                                                                     batch_size,
                                                                                                     num_workers)
            srocc_all[d] = srocc_dataset
            plcc_all[d] = plcc_dataset
            regressors[d] = regressor
            alphas[d] = alpha
            best_worst_results_all[d] = best_worst_results
            print(f"{datetime.now().strftime("%d/%m/%Y %H:%M:%S")} - {dataset_name}:"
                  f"SRCC: {np.median(srocc_dataset['global']):.3f} - PLCC: {np.median(plcc_dataset['global']):.3f}")

        return srocc_all, plcc_all, regressors, alphas, best_worst_results_all

    def compute_metrics(self,
                        dataset: Dataset,
                        num_splits: int,
                        phase: str,
                        alpha: float,
                        grid_search: bool,
                        batch_size: int,
                        num_workers: int,
                        eval_type: str = "scratch") -> Tuple[dict, dict, Ridge, float, dict]:
        """
            Compute the metrics for the given model and dataset. If phase == 'test' and grid_search == True, performs
            a grid search over the validation splits to find the best alpha value for the regression.

            Args:
                dataset (torch.utils.data.Dataset): dataset to test on
                num_splits (int): number of splits
                phase (str): phase of the datasets. Must be in ['val', 'test']
                alpha (float): alpha value to use for regression. During test, if None, performs a grid search
                grid_search (bool): whether to perform a grid search over the validation splits to find the best alpha
                value for the regression
                batch_size (int): batch size
                num_workers (int): number of workers for the dataloaders
                eval_type (str): Whether to test a model trained from scratch or the one pretrained by the authors
                of the ARNIQA paper.

            Returns:
                srocc_dataset (dict): dictionary containing the SROCC results for the dataset
                plcc_dataset (dict): dictionary containing the PLCC results for the dataset
                regressor (Ridge): Ridge regressor
                alpha (float): alpha value used for the regression
                best_worst_results (dict): dictionary containing the best and worst results
            """
        srocc_dataset = {"global": []}
        plcc_dataset = {"global": []}
        best_worst_results = {}  # Best and worst 16 results according to the difference between the predicted and
        # the true MOS
        dist_types = None
        if dataset.is_synthetic:
            dist_types = set(dataset.distortion_types)
            for dist_type in dist_types:
                srocc_dataset[dist_type] = []
                plcc_dataset[dist_type] = []

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

        features, scores = self.get_features_scores(dataloader, eval_type)

        # Perform grid search over the validation splits to find the best alpha value for the regression
        if phase == "test" and grid_search:
            best_alpha = self.alpha_grid_search(dataset=dataset, features=features, scores=scores,
                                                num_splits=num_splits)
        else:
            best_alpha = alpha

        for i in range(num_splits):
            train_indices = dataset.get_split_indices(split=i, phase="train")
            test_indices = dataset.get_split_indices(split=i, phase=phase)

            dist_indices = None
            if dataset.is_synthetic:
                dist_indices = {dist_type: np.where(dataset.distortion_types[test_indices] == dist_type)[0] for
                                dist_type in dist_types}

                # for each index generate 5 indices (one for each crop)
                train_indices = np.repeat(train_indices * 5, 5) + np.tile(np.arange(5), len(train_indices))
                test_indices = np.repeat(test_indices * 5, 5) + np.tile(np.arange(5), len(test_indices))

                train_features = features[train_indices]
                train_scores = scores[train_indices]

                regressor =Ridge(alpha=best_alpha).fit(train_features, train_scores)

                test_features = features[test_indices]
                test_scores = scores[test_indices]
                test_scores = test_scores[::5]  # Scores are repeated for each crop, so we only keep the first one
                orig_test_indices = test_indices[::5] // 5  # Get original indices

                preds = regressor.predict(test_features)
                preds = np.mean(np.reshape(preds, (-1, 5)), axis=1)  # Average the predictions of the 5 crops
                # of the same image

                srocc_dataset["global"].append(stats.spearmanr(preds, test_scores)[0])
                plcc_dataset["global"].append(stats.pearsonr(preds, test_scores)[0])

                if dataset.is_synthetic:
                    for dist_type in dist_types:
                        srocc_dataset[dist_type].append(stats.spearmanr(preds[dist_indices[dist_type]],
                                                                        test_scores[dist_indices[dist_type]])[0])
                        plcc_dataset[dist_type].append(stats.pearsonr(preds[dist_indices[dist_type]],
                                                                      test_scores[dist_indices[dist_type]])[0])

                # Compute best and worst results
                if i == 0:
                    diff = np.abs(preds - test_scores)
                    sorted_diff_indices = np.argsort(diff)
                    best_indices = sorted_diff_indices[:16]
                    worst_indices = sorted_diff_indices[16:][::-1]
                    best_worst_results["best"] = {"images": dataset.images[orig_test_indices[best_indices]],
                                                  "gts": test_scores[best_indices], "preds": preds[best_indices]}
                    best_worst_results["worst"] = {"images": dataset.images[orig_test_indices[worst_indices]],
                                                   "gts": test_scores[worst_indices], "preds": preds[worst_indices]}

            # Train a regressor on the whole dataset for saving purposes
            regressor = Ridge(alpha=best_alpha).fit(features, scores)

            return srocc_dataset, plcc_dataset, regressor, best_alpha, best_worst_results

    def get_features_scores(self,
                            dataloader: DataLoader,
                            eval_type: str = "scratch") -> Tuple[np.ndarray, np.ndarray]:
        """
            Get the features and scores for the given model and dataloader.

            Args:
                dataloader (torch.utils.data.Dataloader): dataloader
                eval_type (str): Whether to test a model trained from scratch or the one pretrained by the authors of
                the ARNIQA paper.

            Returns:
                features (np.ndarray): features
                scores (np.ndarray): ground-truth MOS scores
        """
        feats = np.zeros((0, self.clr.encoder.feat_dim * 2))  # Double the features because of the original and
        # downsampled image
        scores = np.zeros(0)

        for i, batch in enumerate(dataloader):
            img_orig = batch["img"].to(self.device)
            img_ds = batch["img_ds"].to(self.device)
            mos = batch["mos"]

            img_orig = rearrange(img_orig, "b n c h w -> (b n) c h w")
            img_ds = rearrange(img_ds, "b n c h w -> (b n) c h w")
            mos = mos.repeat_interleave(5)  # repeat MOS for each crop

            with torch.cuda.amp.autocast(), torch.no_grad():
                if eval_type == "scratch":
                    f_orig, _ = self.clr(img_orig)
                    f_ds, _ = self.clr(img_ds)
                    f = torch.hstack((f_orig, f_ds))
                elif eval_type == "arniqa":
                    _, f = self.clr(img_orig, img_ds, return_embedding=True)

            feats = np.concatenate((feats, f.cpu().numpy()), axis=0)
            scores = np.concatenate((scores, mos.numpy()), axis=0)

        return feats, scores

    def alpha_grid_search(self,
                          dataset: Dataset,
                          features: np.ndarray,
                          scores: np.ndarray,
                          num_splits: int) -> float:
        """
            Perform a grid search over the validation splits to find the best alpha value for the regression based on
            the SROCC metric. The grid search is performed over the range [1-e3, 1e3, 100].

            Args:
                dataset (Dataset): dataset to use
                features (np.ndarray): features extracted with the model to test
                scores (np.ndarray): ground-truth MOS scores
                num_splits (int): number of splits to use

            Returns:
                alpha (float): best alpha value
        """
        grid_search_range = [1e-3, 1e3, 100]
        alphas = np.geomspace(*grid_search_range, endpoint=True)
        srocc_all = [[] for _ in range(len(alphas))]

        for i in range(num_splits):
            train_indices = dataset.get_split_indices(split=i, phase="train")
            val_indices = dataset.get_split_indices(split=i, phase="val")

            # for each index generate 5 indices (one for each crop)
            train_indices = np.repeat(train_indices * 5, 5) + np.tile(np.arange(5), len(train_indices))
            val_indices = np.repeat(val_indices * 5, 5) + np.tile(np.arange(5), len(val_indices))

            train_features = features[train_indices]
            train_scores = scores[train_indices]

            val_features = features[val_indices]
            val_scores = scores[val_indices]
            val_scores = val_scores[::5]  # Scores are repeated for each crop, so we only keep the first one

            for idx, alpha in enumerate(alphas):
                regressor = Ridge(alpha=alpha).fit(train_features, train_scores)
                preds = regressor.predict(val_features)
                preds = np.mean(np.reshape(preds, (-1, 5)), axis=1)  # Average the predictions of the 5 crops
                # of the same image
                srocc_all[idx].append(stats.spearmanr(preds, val_scores)[0])

        srocc_all_median = [np.median(srocc) for srocc in srocc_all]
        srocc_all_median = np.array(srocc_all_median)
        best_alpha_idx = np.argmax(srocc_all_median)
        best_alpha = alphas[best_alpha_idx]

        return best_alpha
