import numpy as np
import torchvision
from src.datasets.contrastive import ContrastiveDataset


def test_contrastive_size() -> None:
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    dataset = torchvision.datasets.CIFAR10(
        root="~/pytorch_datasets", download=True, train=False
    )

    rnd = np.random.RandomState(7)
    for num_n_pairs_per_sample in range(1, 2):
        train_set = ContrastiveDataset(
            dataset,
            rnd=rnd,
            num_n_pairs_per_sample=num_n_pairs_per_sample,
            transform=transform,
        )

        assert len(train_set) == len(dataset) * num_n_pairs_per_sample
        for i in range(2):
            assert train_set[0][i].shape == (3, 32, 32)
