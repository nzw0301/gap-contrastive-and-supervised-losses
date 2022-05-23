from collections import Counter
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import torch.utils.data
import torchtext.vocab
from torchtext.data.utils import get_tokenizer
from torchtext.utils import download_from_url, extract_archive
from torchtext.vocab import Vocab

from .utils import ClassDownSampler, _train_val_split

_URL = "https://nlp.cs.princeton.edu/CURL/Wiki-3029.tar.gz"


class Wiki3029(torch.utils.data.Dataset):
    def __init__(self, root: str) -> None:
        dataset_tar = download_from_url(_URL, root=root)
        extracted_files = extract_archive(dataset_tar)
        self.data = []
        self.targets = []

        for y, p in enumerate(sorted(extracted_files)):
            with open(p) as f:
                for line in f:
                    self.data.append(line.strip())
                    self.targets.append(y)

    def __len__(self) -> int:
        return len(self.targets)  # 200 * 3029

    def __getitem__(self, index) -> Tuple[str, int]:
        return self.data[index], self.targets[index]


class IntWiki3029(torch.utils.data.Dataset):
    def __init__(
        self, dataset: Wiki3029, fn: Callable, vocab: torchtext.vocab.Vocab
    ) -> None:
        self.data = [fn(x) for x in dataset.data]
        self.targets = dataset.targets
        self.vocab = vocab

    def __getitem__(self, index: int) -> Tuple[List[int], int]:
        return self.data[index], self.targets[index]

    def __len__(self) -> int:
        return len(self.data)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)


def get_train_val_test_datasets(
    rnd: np.random.RandomState,
    root="~/pytorch_datasets",
    validation_ratio: float = 0.1,
    test_ratio: float = 0.2,
    min_freq: int = 5,
    train_size: Union[int, float] = 3029,
) -> Tuple[IntWiki3029, Optional[IntWiki3029], Optional[IntWiki3029]]:
    """
    By following Ash et al.'s train/val split and
    https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html#prepare-data-processing-pipelines

    :param rnd: `np.random.RandomState` instance.
    :param validation_ratio: The ratio of validation data. If this value is `0.`, returned `val_set` is `None`.
    :param test_ratio: The ratio of test data. If this value is `0.`, returned `test_set` is `None`.
    :param root: Path to save data.
    :param min_freq: The minimal frequency of words used in the training.
    :param train_size: If the value is int, it represents the number of classes.

    :return: Tuple of (train dataset, val dataset, test dataset).
    """

    dataset = Wiki3029(root=root)

    # create validation split
    if validation_ratio > 0.0:
        train_set, val_set = _train_val_split(
            rnd=rnd,
            train_dataset=dataset,
            validation_ratio=validation_ratio + test_ratio,
        )
        if test_ratio > 0.0:
            # in this split
            # `validation_ratio` represents the ratio of test dataset.
            val_set, test_set = _train_val_split(
                rnd=rnd,
                train_dataset=val_set,
                validation_ratio=test_ratio / (validation_ratio + test_ratio),
            )
        else:
            test_set = None
    else:
        val_set = None

        if test_ratio > 0.0:  # no validation set, but test data exists
            train_set, test_set = _train_val_split(
                rnd=rnd, train_dataset=dataset, validation_ratio=test_ratio
            )

        else:  # neither validation set nor test set
            train_set = dataset
            test_set = None

    down_sampler = ClassDownSampler(rnd=rnd, sklearn_train_size=train_size)

    train_set = down_sampler.fit_transform(train_set)

    if val_set is not None:
        val_set = down_sampler.transform(val_set)

    if test_set is not None:
        test_set = down_sampler.transform(test_set)

    tokenizer = get_tokenizer("basic_english")

    counter = Counter()
    for line, _ in train_set:
        counter.update(tokenizer(line))

    vocab = Vocab(counter, specials=["<unk>"], min_freq=min_freq, specials_first=False)

    text_pipeline = lambda x: [vocab[token] for token in tokenizer(x)]

    train_set = IntWiki3029(train_set, text_pipeline, vocab)

    if val_set is not None:
        val_set = IntWiki3029(val_set, text_pipeline, vocab)

    if test_set is not None:
        test_set = IntWiki3029(test_set, text_pipeline, vocab)

    return train_set, val_set, test_set


def collate_supervised_batch(batch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    mainly from
    https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html#generate-data-batch-and-iterator
    for torchtext==0.9.0
    """

    label_list, text_list, offsets = [], [], [0]
    for (int_words, label) in batch:
        label_list.append(label)
        processed_text = torch.tensor(int_words, dtype=torch.int64)
        text_list.append(processed_text)
        offsets.append(processed_text.size(0))

    label_list = torch.tensor(label_list, dtype=torch.int64)
    text_list = torch.cat(text_list)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)

    return text_list, label_list, offsets


def collate_contrastive_batch(
    batch,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    mainly from
    https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html#generate-data-batch-and-iterator
    for torchtext==0.9.0
    """

    anchor_text_list, anchor_offsets = [], [0]
    positive_text_list, positive_offsets = [], [0]

    for (anchor_int_words, positive_int_words) in batch:
        anchor_processed_text = torch.tensor(anchor_int_words, dtype=torch.int64)
        anchor_text_list.append(anchor_processed_text)
        anchor_offsets.append(anchor_processed_text.size(0))

        positive_processed_text = torch.tensor(positive_int_words, dtype=torch.int64)
        positive_text_list.append(positive_processed_text)
        positive_offsets.append(positive_processed_text.size(0))

    anchor_text_list = torch.cat(anchor_text_list)
    anchor_offsets = torch.tensor(anchor_offsets[:-1]).cumsum(dim=0)

    positive_text_list = torch.cat(positive_text_list)
    positive_offsets = torch.tensor(positive_offsets[:-1]).cumsum(dim=0)

    return anchor_text_list, anchor_offsets, positive_text_list, positive_offsets
