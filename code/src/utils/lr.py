import numpy as np
from omegaconf import OmegaConf


def calculate_initial_lr(cfg: OmegaConf, mini_batch_size: int) -> float:
    """
    Proposed initial learning rates by SimCLR paper.
    Note: SimCLR paper says squared learning rate is better when the size of mini-batches is small.
    :return: Initial learning rate whose type is float.
    """

    if cfg["lr_scheduler"]["linear_schedule"]:
        scaled_lr = cfg["optimizer"]["lr"] * mini_batch_size / 256.0
    else:
        scaled_lr = cfg["optimizer"]["lr"] * np.sqrt(mini_batch_size)

    return scaled_lr


def calculate_lr_list(cfg: OmegaConf, batch_size: int, num_iters: int) -> np.ndarray:
    """
    scaling + linear warmup + cosine annealing without restart
    https://github.com/facebookresearch/swav/blob/master/main_swav.py#L178-L182
    Note that the first lr is 0.

    :param cfg: Hydra's config
    :param batch_size: the size of mini-batch
    :param num_iters: the number of iterations, in other words, the number of batches per epoch.

    :return: np.ndarray of learning rates for all steps.
    """
    base_lr = calculate_initial_lr(cfg, batch_size)

    warmup_lr_schedule = np.linspace(
        0, base_lr, num_iters * cfg["lr_scheduler"]["warmup_epochs"]
    )
    iters = np.arange(
        num_iters * (cfg["epochs"] - cfg["lr_scheduler"]["warmup_epochs"])
    )
    cosine_lr_schedule = np.array(
        [
            0.5
            * base_lr
            * (
                1
                + np.cos(
                    np.pi
                    * t
                    / (
                        num_iters
                        * (cfg["epochs"] - cfg["lr_scheduler"]["warmup_epochs"])
                    )
                )
            )
            for t in iters
        ]
    )

    return np.concatenate((warmup_lr_schedule, cosine_lr_schedule))
