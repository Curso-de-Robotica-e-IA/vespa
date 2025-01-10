from torch.optim import Adam, SGD, RMSprop, Adagrad

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


def custom_collate_fn(batch):
    """Custom collate function for DataLoader."""
    return tuple(zip(*batch))

