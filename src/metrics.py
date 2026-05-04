import torch


def si_rmse(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Scale-invariant RMSE as defined in the competition."""
    valid = (target > eps) & (pred > eps)
    d = torch.log(pred[valid]) - torch.log(target[valid])
    alpha = torch.mean(d)
    return torch.sqrt(torch.mean((d - alpha) ** 2))
