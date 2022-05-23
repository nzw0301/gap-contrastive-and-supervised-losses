import numpy as np
import torchvision
from src.datasets.utils import ClassDownSampler
from src.datasets.wiki3029 import Wiki3029


def test_class_down_sampling_trivial_cifar10() -> None:
    rnd = np.random.RandomState(7)

    # trivial case
    dataset = torchvision.datasets.CIFAR10(root="~/pytorch_datasets", download=True)

    down_sampler = ClassDownSampler(rnd, 1.0)
    down_sampler.fit(dataset)
    down_sampled_dataset = down_sampler.transform(dataset)
    num_samples = len(dataset)
    num_classes = 10

    assert len(down_sampled_dataset) == num_samples
    assert len(down_sampled_dataset.targets) == num_samples
    assert len(np.unique(down_sampled_dataset.targets)) == num_classes

    # trivial case: int
    dataset = torchvision.datasets.CIFAR10(root="~/pytorch_datasets", download=True)

    down_sampler = ClassDownSampler(rnd, 10)
    down_sampler.fit(dataset)
    down_sampled_dataset = down_sampler.transform(dataset)
    num_samples = len(dataset)
    num_classes = 10

    assert len(down_sampled_dataset) == num_samples
    assert len(down_sampled_dataset.targets) == num_samples
    assert len(np.unique(down_sampled_dataset.targets)) == num_classes

    # trivial case: pass all classes to target_classes
    down_sampler = ClassDownSampler(rnd, target_classes=np.arange(num_classes))
    down_sampler.fit(dataset)
    down_sampled_dataset = down_sampler.transform(dataset)
    num_samples = len(dataset)

    assert len(down_sampled_dataset) == num_samples
    assert len(down_sampled_dataset.targets) == num_samples
    assert len(np.unique(down_sampled_dataset.targets)) == num_classes


def test_class_down_sampling_trivial_wiki3029() -> None:
    rnd = np.random.RandomState(7)

    # trivial case
    dataset = Wiki3029(root="~/pytorch_datasets")

    down_sampler = ClassDownSampler(rnd, 1.0)
    down_sampler.fit(dataset)
    down_sampled_dataset = down_sampler.transform(dataset)

    num_samples = len(dataset)
    num_classes = 3029

    assert len(down_sampled_dataset) == num_samples
    assert len(down_sampled_dataset.targets) == num_samples
    assert len(np.unique(down_sampled_dataset.targets)) == num_classes

    down_sampler = ClassDownSampler(rnd, num_classes)
    down_sampler.fit(dataset)
    down_sampled_dataset = down_sampler.transform(dataset)

    assert len(down_sampled_dataset) == num_samples
    assert len(down_sampled_dataset.targets) == num_samples
    assert len(np.unique(down_sampled_dataset.targets)) == num_classes

    down_sampler = ClassDownSampler(rnd, target_classes=np.arange(num_classes))
    down_sampler.fit(dataset)
    down_sampled_dataset = down_sampler.transform(dataset)
    num_samples = len(dataset)

    assert len(down_sampled_dataset) == num_samples
    assert len(down_sampled_dataset.targets) == num_samples
    assert len(np.unique(down_sampled_dataset.targets)) == num_classes


def test_class_down_sampling_cifar10() -> None:
    rnd = np.random.RandomState(7)

    # float version
    dataset = torchvision.datasets.CIFAR10(root="~/pytorch_datasets", download=True)
    train_size = 0.2
    down_sampler = ClassDownSampler(rnd, train_size)
    down_sampled_dataset = down_sampler.fit_transform(dataset)

    assert len(down_sampled_dataset.data) == 2 * 5000
    assert len(down_sampled_dataset.targets) == 2 * 5000
    assert len(np.unique(down_sampled_dataset.targets)) == 2

    # sampled id is relabelled, so the class starts 0
    assert 0 in np.unique(down_sampled_dataset.targets)
    assert 1 in np.unique(down_sampled_dataset.targets)

    # original dataset is not changed
    assert len(dataset.data) == 10 * 5000

    # int version
    train_size = 2
    down_sampler = ClassDownSampler(rnd, train_size)
    down_sampled_dataset = down_sampler.fit_transform(dataset)
    assert len(down_sampled_dataset.data) == train_size * 5000
    assert len(down_sampled_dataset.targets) == train_size * 5000
    assert len(np.unique(down_sampled_dataset.targets)) == train_size

    # sampled id is relabelled, so the class starts 0
    assert 0 in np.unique(down_sampled_dataset.targets)
    assert 1 in np.unique(down_sampled_dataset.targets)

    # original dataset is not changed
    assert len(dataset.data) == 10 * 5000

    # target_classes
    down_sampler = ClassDownSampler(rnd, target_classes=[7, 1])
    down_sampled_dataset = down_sampler.fit_transform(dataset)

    assert len(down_sampled_dataset) == 2 * 5000
    assert len(down_sampled_dataset.targets) == 2 * 5000
    assert len(np.unique(down_sampled_dataset.targets)) == 2

    # sampled id is relabelled, so the class starts 0
    assert 0 in np.unique(down_sampled_dataset.targets)
    assert 1 in np.unique(down_sampled_dataset.targets)

    # original dataset is not changed
    assert len(dataset.data) == 10 * 5000


def test_class_down_sampling_wikipedia() -> None:
    rnd = np.random.RandomState(7)

    # float version
    dataset = Wiki3029(root="~/pytorch_datasets")
    num_classes = 3029
    num_samples_per_class = 200

    train_size = 0.2
    num_samples = int(num_classes * train_size) * num_samples_per_class
    down_sampler = ClassDownSampler(rnd, train_size)
    down_sampled_dataset = down_sampler.fit_transform(dataset)

    assert len(down_sampled_dataset.data) == num_samples
    assert len(down_sampled_dataset.targets) == num_samples
    assert len(np.unique(down_sampled_dataset.targets)) == int(num_classes * train_size)

    # sampled id is relabelled, so the class starts 0
    for i in range(int(num_classes * train_size)):
        assert i in np.unique(down_sampled_dataset.targets)

    # original dataset is not changed
    assert len(dataset.data) == num_classes * num_samples_per_class

    # int version
    train_size = 100  # the number of used classes
    num_samples = train_size * num_samples_per_class
    down_sampler = ClassDownSampler(rnd, train_size)
    down_sampled_dataset = down_sampler.fit_transform(dataset)

    assert len(down_sampled_dataset.data) == num_samples
    assert len(down_sampled_dataset.targets) == num_samples
    assert len(np.unique(down_sampled_dataset.targets)) == train_size

    # original dataset is not changed
    assert len(dataset.data) == num_classes * num_samples_per_class

    # target classes version
    down_sampler = ClassDownSampler(rnd, target_classes=[7, 1])
    down_sampled_dataset = down_sampler.fit_transform(dataset)
    num_expected_classes = 2

    assert len(down_sampled_dataset) == num_expected_classes * num_samples_per_class
    assert (
        len(down_sampled_dataset.targets)
        == num_expected_classes * num_samples_per_class
    )
    assert len(np.unique(down_sampled_dataset.targets)) == num_expected_classes

    # sampled id is relabelled, so the class starts 0
    for i in range(num_expected_classes):
        assert i in np.unique(down_sampled_dataset.targets)

    # original dataset is not changed
    assert len(dataset.data) == num_classes * num_samples_per_class
