import torch


class DownstreamDataset(torch.utils.data.Dataset):
    def __init__(self, data, targets) -> None:
        assert len(data) == len(
            targets
        ), "the numbers of samples and the number of labeled differ."

        self.data = data
        self.targets = targets

    def __getitem__(self, index: int) -> tuple:
        return self.data[index], self.targets[index]

    def __len__(self) -> int:
        return len(self.data)
