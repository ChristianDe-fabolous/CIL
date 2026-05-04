import torch


def silog_loss(pred: torch.Tensor, target: torch.Tensor, lambda_: float = 0.5, eps: float = 1e-6) -> torch.Tensor:
    valid = (target > eps) & (pred > eps)
    d = torch.log(pred[valid]) - torch.log(target[valid])
    loss = torch.mean(d ** 2) - lambda_ * torch.mean(d) ** 2
    return torch.sqrt(loss.clamp(min=1e-8))
