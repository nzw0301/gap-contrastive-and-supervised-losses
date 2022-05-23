import torch
from src.losses.contrastive_loss import ContrastiveLoss


def test_class_down_sampling_trivial_cifar10() -> None:

    device = torch.device("cpu")
    dim = 12
    num_batch_size = 32
    num_negatives = 64

    contrastive_loss = ContrastiveLoss(device=device, num_negatives=num_negatives)

    anchor = torch.rand(num_batch_size, dim)
    positive = torch.rand(num_batch_size, dim)
    contrastive_loss(anchor, positive)
