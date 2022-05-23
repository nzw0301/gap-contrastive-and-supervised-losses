from typing import Optional, Tuple, Union

import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, CIFAR100

from .utils import _train_val_split


def get_train_val_test_datasets(
    rnd: np.random.RandomState,
    root="~/pytorch_datasets",
    validation_ratio=0.05,
    dataset_name="cifar100",
    normalize=False,
) -> Tuple[
    Optional[Union[CIFAR10, CIFAR100]],
    Optional[Union[CIFAR10, CIFAR100]],
    Optional[Union[CIFAR10, CIFAR100]],
]:
    """
    Create CIFAR-10/100 train/val/test data loaders

    :param rnd: `np.random.RandomState` instance.
    :param validation_ratio: The ratio of validation data. If this value is `0.`, returned `val_set` is `None`.
    :param root: Path to save data.
    :param dataset_name: The name if dataset. cifar10 or cifar100
    :param normalize: flag to perform channel-wise normalization as pre-processing.

    :return: Tuple of (train, val, test).
    """

    if dataset_name not in {"cifar10", "cifar100"}:
        raise ValueError

    transform = transforms.Compose([transforms.ToTensor(),])

    if dataset_name == "cifar10":
        DataSet = CIFAR10
    else:
        DataSet = CIFAR100

    train_dataset = DataSet(root=root, train=True, download=True, transform=transform)

    # create validation split
    if validation_ratio > 0.0:
        train_dataset, val_dataset = _train_val_split(
            rnd=rnd, train_dataset=train_dataset, validation_ratio=validation_ratio
        )
    else:
        val_dataset = None

    if normalize:
        # create a transform to do pre-processing
        train_loader = DataLoader(
            train_dataset, batch_size=len(train_dataset), shuffle=False
        )

        data = iter(train_loader).next()
        dim = [0, 2, 3]
        mean = data[0].mean(dim=dim).numpy()
        std = data[0].std(dim=dim).numpy()
        # end of creating a transform to do pre-processing

        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std),]
        )

        train_dataset.transform = transform

    if val_dataset is not None:
        val_dataset.transform = transform

    test_dataset = DataSet(root=root, train=False, download=True, transform=transform)

    return train_dataset, val_dataset, test_dataset
