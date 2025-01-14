import torch
from torch.optim import Adam, SGD, RMSprop, Adagrad
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_curve, auc
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt

class ModelUtils:
    @staticmethod
    def print_model_summary(model):
        """Print a summary of the model."""
        print(model)


    @staticmethod
    def count_trainable_parameters(model):
        """Count the number of trainable parameters in the model."""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


    @staticmethod
    def freeze_backbone(model):
        """Freeze the backbone to prevent updates during training."""
        for param in model.backbone.parameters():
            param.requires_grad = False


    @staticmethod
    def unfreeze_backbone(model):
        """Unfreeze the backbone to allow updates during training."""
        for param in model.backbone.parameters():
            param.requires_grad = True


    @staticmethod
    def adjust_learning_rate(optimizer, new_lr):
        """Adjust the learning rate of the optimizer."""
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
        print(f'Learning rate adjusted to {new_lr:.6f}')


    @staticmethod
    def calculate_metrics(y_true, y_pred):
        """Calculate common metrics for model evaluation."""
        return {
            "precision": precision_score(y_true, y_pred, average="weighted"),
            "recall": recall_score(y_true, y_pred, average="weighted"),
            "f1_score": f1_score(y_true, y_pred, average="weighted"),
            "accuracy": accuracy_score(y_true, y_pred),
        }


    @staticmethod
    def perform_ttest(group1, group2):
        """Perform a two-sample t-test to compare means of two groups."""
        t_stat, p_value = ttest_ind(group1, group2)
        return {"t_statistic": t_stat, "p_value": p_value}


    @staticmethod
    def plot_roc_curve(y_true, y_score):
        """Plot the ROC curve for a binary classification model."""
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic")
        plt.legend(loc="lower right")
        plt.show()


    @staticmethod
    def save_model(model, path):
        """Save the model to the specified path."""
        torch.save(model.state_dict(), path)


    @staticmethod
    def load_model(model, path, device="cpu"):
        """Load a model from a specified path."""
        model.load_state_dict(torch.load(path, map_location=device))
        return model


    @staticmethod
    def configure_optimizer(model, optimizer_name: str, lr: float = 0.0001, weight_decay: float = 0.0001):
        """
        Configure the optimizer for a given model.

        Args:
            model (torch.nn.Module): The model to optimize.
            optimizer_name (str): Name of the optimizer ('adam', 'sgd', etc.).
            lr (float): Learning rate. Defaults to 0.0001.
            weight_decay (float): Weight decay (L2 penalty). Defaults to 0.0001.

        Returns:
            torch.optim.Optimizer: Configured optimizer instance.
        """
        optimizer_name = optimizer_name.lower()
        if optimizer_name == 'adam':
            return Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'sgd':
            return SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
        elif optimizer_name == 'rmsprop':
            return RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'adagrad':
            return Adagrad(model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")


    @staticmethod
    def custom_collate_fn(batch):
        """
        Custom collate function for DataLoader.

        Args:
            batch: List of samples from the dataset.

        Returns:
            Tuple: Collated batch for the DataLoader.
        """
        return tuple(zip(*batch))
