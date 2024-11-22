import torch


def wasserstein_loss(estim: torch.Tensor, is_real: bool = True) -> torch.Tensor:
    """
    Calculates the Wasserstein loss

    Parameters
    ----------
    estim: torch.Tensor
        The estimated values
    is_real: bool
        Whether the values are real or fake

    Returns
    -------
    torch.Tensor
        The calculated loss
    """
    if is_real:
        return -torch.mean(estim)
    else:
        return torch.mean(estim)
