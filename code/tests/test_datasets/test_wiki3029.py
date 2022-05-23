import numpy as np
import pytest
from src.datasets.wiki3029 import get_train_val_test_datasets

train_sizes = [3029, 200]


@pytest.mark.parametrize("train_size", train_sizes)
def test_trivial(train_size: int) -> None:
    validation_ratio = 0.0
    test_ratio = 0.0
    expected_total_samples = train_size * 200

    rnd = np.random.RandomState(7)
    train, val, test = get_train_val_test_datasets(
        rnd,
        validation_ratio=validation_ratio,
        test_ratio=test_ratio,
        train_size=train_size,
    )

    assert len(train) == expected_total_samples
    assert len(np.unique(train.targets)) == train_size
    assert val is None
    assert test is None


@pytest.mark.parametrize("train_size", train_sizes)
def test_trivial(train_size: int) -> None:
    validation_ratio = 0.1
    test_ratio = 0.2
    expected_total_samples = train_size * 200

    rnd = np.random.RandomState(7)
    train, val, test = get_train_val_test_datasets(
        rnd,
        validation_ratio=validation_ratio,
        test_ratio=test_ratio,
        train_size=train_size,
    )

    assert len(train) == pytest.approx(expected_total_samples * 0.7, abs=2)
    assert len(np.unique(train.targets)) == train_size
    assert len(train) + len(val) + len(test) == expected_total_samples
    assert len(val) == pytest.approx(expected_total_samples * 0.1, abs=2)
    assert len(test) == pytest.approx(expected_total_samples * 0.2, abs=2)
