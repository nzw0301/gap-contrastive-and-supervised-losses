import copy
import logging
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from sklearn.model_selection import train_test_split as sk_train_val_split


def _train_val_split(
    rnd: np.random.RandomState,
    train_dataset: torch.utils.data.Dataset,
    validation_ratio: float = 0.05,
) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """
    Apply sklearn's `train_val_split` function to PyTorch's dataset instance.

    :param rnd: `np.random.RandomState` instance.
    :param train_dataset: Training set. This is an instance of PyTorch's dataset.
    :param validation_ratio: The ratio of validation data.

    :return: Tuple of training set and validation set.
    """

    x_train, x_val, y_train, y_val = sk_train_val_split(
        train_dataset.data,
        train_dataset.targets,
        test_size=validation_ratio,
        random_state=rnd,
        stratify=train_dataset.targets,
    )

    val_dataset = copy.deepcopy(train_dataset)

    train_dataset.data = x_train
    train_dataset.targets = y_train

    val_dataset.data = x_val
    val_dataset.targets = y_val

    return train_dataset, val_dataset


class ClassDownSampler(object):
    """
    Remove a part of classes from the dataset.
    `sklearn_train_size` is passed to `train_size` of `sklearn.model_selection.train_test_split`.
    If `target_ids` is not None, `sklearn_train_size` is ignored.

    We should pass `target_classes` for validation and test.

    Note that the interface looks sklearn's fit and transform, but this class does not inherit sklearn's class.
    """

    def __init__(
        self,
        rnd: np.random.RandomState,
        sklearn_train_size: Optional[Union[float, int]] = None,
        target_classes: Optional[Sequence[Any]] = None,
    ):
        """
        :param rnd: for reproducibility to split train/val split.
        :param: sklearn_train_size: used as `sklearn.model_selection.train_val_split`'s `train_size` argument.
            This value is not used if `target_classes` is specified.
        :param target_classes: the pre-defined used classes. If a class is not not in it, the samples are removed.
            If this value is not None, `sklearn_train_size` will be ignored.
        """

        assert not (sklearn_train_size is None and target_classes is None)

        self.rnd = rnd
        self.train_size = sklearn_train_size
        self.target_classes = target_classes

        self.relabeling_dict: Optional[Dict[Any, int]] = None

    @staticmethod
    def _check_unique_target_classes(target_classes: Optional[Sequence[Any]]) -> bool:
        return target_classes is None or len(np.unique(target_classes)) == len(
            target_classes
        )

    def _check_trivial(self, classes: Sequence[Any]) -> bool:
        """
        :param classes: the target classes of samples.

        Check whether or not `ClassDownSampler` performs down-sampling of classes.
        """
        return (
            (isinstance(self.train_size, int) and self.train_size == len(classes))
            or (isinstance(self.train_size, float) and self.train_size == 1.0)
            or (
                self.target_classes is not None
                and len(self.target_classes) == len(classes)
            )
        )

    def fit(self, dataset: torch.utils.data.Dataset) -> None:
        """
        Performing the down-sampling of classes.
        """
        classes = np.unique(dataset.targets)

        if self._check_trivial(classes):
            logging.warning("The target classes are trivial. No down-sampling performs")
            self.relabeling_dict = {c: c for c in classes}
            return

        # sampling target classes if we do not specify `target_classes`
        if self.target_classes is None:
            target_classes_ids, _ = sk_train_val_split(
                classes, random_state=self.rnd, train_size=self.train_size
            )
            self.target_classes = classes[target_classes_ids]

        self.relabeling_dict = {
            c: new_label for new_label, c in enumerate(self.target_classes)
        }

    def fit_transform(
        self, dataset: torch.utils.data.Dataset
    ) -> torch.utils.data.Dataset:
        """
        Performing the down-sampling of classes by calling `fit`,
        then remove samples and re-label them based on the `fit`'s results.
        """
        self.fit(dataset)
        return self.transform(dataset)

    def transform(self, dataset: torch.utils.data.Dataset) -> torch.utils.data.Dataset:
        """
        Remove samples and re-label them based on the `fit`'s results.
        """
        assert self.relabeling_dict is not None, "`fit` is not called yet."

        _dataset = copy.deepcopy(dataset)  # to avoid overwrite the original dataset

        # trivial case: use all data
        if self._check_trivial(np.unique(dataset.targets)):
            return _dataset

        down_sampled_data = []
        down_sampled_targets = []
        for x, y in zip(dataset.data, dataset.targets):
            if y in self.relabeling_dict:
                down_sampled_targets.append(self.relabeling_dict[y])
                down_sampled_data.append(x)

        _dataset.data = down_sampled_data
        _dataset.targets = down_sampled_targets

        return _dataset
