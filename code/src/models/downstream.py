import torch
from torch.utils.data import Dataset


class NonLinearClassifier(torch.nn.Module):
    def __init__(
        self, num_features: int = 128, num_hidden: int = 128, num_classes: int = 10
    ) -> None:
        super(NonLinearClassifier, self).__init__()

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(num_features, num_hidden),
            torch.nn.BatchNorm1d(num_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(num_hidden, num_classes, bias=False),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Return Unnormalized probabilities.

        :param inputs: Mini-batches of feature representation.
        :return: Unnormalized probabilities.
        """

        return self.classifier(inputs)  # N x num_classes


class LinearClassifier(torch.nn.Module):
    def __init__(self, num_features: int = 128, num_classes: int = 10) -> None:
        """
        Linear classifier for linear evaluation protocol.

        :param num_features: The dimensionality of feature representation
        :param num_classes: The number of supervised class
        """
        super(LinearClassifier, self).__init__()
        self.classifier = torch.nn.Linear(num_features, num_classes, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Return Unnormalized probabilities

        :param inputs: Mini-batches of feature representation.
        :return: Unnormalized probabilities in torch.Tensor.
        """
        return self.classifier(inputs)  # N x num_classes


class MeanClassifier(torch.nn.Module):
    def __init__(self, weights: torch.Tensor) -> None:
        """
        :param weights: The pre-computed weights of the classifier.
        """
        super(MeanClassifier, self).__init__()
        self.weights = weights  # d x num_classes

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.matmul(inputs, self.weights)  # N x num_classes

    @staticmethod
    def create_weights(
        dataset: Dataset, num_classes: int, normalize: bool = False
    ) -> torch.Tensor:
        """
        :param dataset: Dataset of feature representation to create weights.
        :param num_classes: The number of classes.
        :param normalize: whether or not to perform normalization after feature extraction.

        :return: FloatTensor contains weights.
        """

        X = dataset.data
        Y = dataset.targets

        weights = []
        for k in range(num_classes):
            ids = torch.where(Y == k)[0]
            if normalize:
                mean_vector = torch.mean(
                    torch.nn.functional.normalize(X[ids], p=2, dim=1), dim=0
                )
            else:
                mean_vector = torch.mean(X[ids], dim=0)
            weights.append(mean_vector)

        weights = torch.stack(weights, dim=1)  # d x num_classes
        return weights
