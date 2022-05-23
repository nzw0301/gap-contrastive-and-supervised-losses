from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image


def _get_class2samples(
    dataset: torch.utils.data.Dataset, num_classes: int
) -> List[List]:
    """
    Create list such that each index is corresponding to the class_id, and each element is a tensor of data.
    :param dataset: CIFAR10/100/IntWiki3029 dataset's instance.
    :param num_classes: The number of classes.
    :return: list of list that contains tensor per class.
    """
    label2samples = [[] for _ in range(num_classes)]

    for sample, label in zip(dataset.data, dataset.targets):
        label2samples[label].append(sample)

    return label2samples


class ContrastiveDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        rnd: np.random.RandomState,
        num_n_pairs_per_sample: int = 1,
        transform: Optional[Callable] = None,
        is_image_dataset: bool = True,
    ) -> None:
        """
        Contrastive Dataset.

        :param dataset: Instance of torchvision.datasets.CIFAR10, torchvision.datasets.CIFAR100, IntWiki3029.
        :param rnd: random state for reproducibility to generate anchor/positive pairs
        :param num_n_pairs_per_sample: The number of N-pairs per sample.
        :param transform: transform for input samples
        :param is_image_dataset: flag for internal transform.
        """

        self.rnd: np.random.RandomState = rnd
        self.num_n_pairs_per_sample = num_n_pairs_per_sample
        self.transform = transform
        self.is_image_dataset = is_image_dataset

        self.num_classes = len(np.unique(dataset.targets))
        self.data: list = self._build_n_pair_dataset(dataset)

    def _build_n_pair_dataset(self, dataset: torch.utils.data.Dataset) -> List:
        n_pairs_list = []

        class_id2samples = _get_class2samples(dataset, self.num_classes)
        for class_id in range(self.num_classes):
            N = len(class_id2samples[class_id])
            for anchor_id in range(N):
                for _ in range(self.num_n_pairs_per_sample):
                    n_pairs = list()
                    n_pairs.append(class_id2samples[class_id][anchor_id])
                    positive_id = self.rnd.randint(low=0, high=N)
                    # avoid anchor == positive
                    while anchor_id == positive_id:
                        positive_id = self.rnd.randint(low=0, high=N)
                    n_pairs.append(class_id2samples[class_id][positive_id])

                    n_pairs_list.append(n_pairs)

        return n_pairs_list

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        anchor, positive = self.data[index]

        if self.is_image_dataset:
            anchor = Image.fromarray(anchor)
            positive = Image.fromarray(positive)

        if self.transform is not None:
            anchor = self.transform(anchor)
            positive = self.transform(positive)

        if isinstance(anchor, np.ndarray):
            anchor = torch.Tensor(anchor)
            positive = torch.Tensor(positive)

        return anchor, positive

    def __len__(self) -> int:
        return len(self.data)


def convert_dataset_to_contrastive_dataset(
    train_dataset: Optional[torch.utils.data.Dataset],
    val_dataset: Optional[torch.utils.data.Dataset],
    rnd: np.random.RandomState,
    train_num_n_pairs_per_sample: int,
    val_num_n_pairs_per_sample: int,
    is_image_dataset: bool = True,
) -> Tuple[Optional[ContrastiveDataset], Optional[ContrastiveDataset]]:
    """
    Create the contrastive datasets from supervised datasets.
    :param train_dataset: the dataset to create training contrastive dataset.
    :param val_dataset: the dataset to create validation contrastive dataset.
    :param rnd: for reproducibility to generate contrastive pairs
    :param train_num_n_pairs_per_sample: the number of pairs per sample for training contrastive dataset
    :param val_num_n_pairs_per_sample: the number of pairs per sample for validation  contrastive dataset
    :param is_image_dataset: flag of image dataset or not. If not, we assume that  the dataset contains numpy array.
    """

    if train_dataset is not None:
        train_dataset = ContrastiveDataset(
            train_dataset,
            rnd=rnd,
            num_n_pairs_per_sample=train_num_n_pairs_per_sample,
            transform=train_dataset.transform
            if hasattr(train_dataset, "transform")
            else None,
            is_image_dataset=is_image_dataset,
        )

    if val_dataset is not None:
        val_dataset = ContrastiveDataset(
            val_dataset,
            rnd=rnd,
            num_n_pairs_per_sample=val_num_n_pairs_per_sample,
            transform=val_dataset.transform
            if hasattr(val_dataset, "transform")
            else None,
            is_image_dataset=is_image_dataset,
        )

    return train_dataset, val_dataset
