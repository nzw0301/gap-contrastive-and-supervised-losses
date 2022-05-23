from collections import OrderedDict

import torch


class SupervisedModel(torch.nn.Module):
    def __init__(
        self, num_embeddings: int, embedding_dim: int = 10, num_classes: int = 3029
    ) -> None:
        """
        :param num_embeddings: The size of vocabulary
        :param embedding_dim: the dimensionality of feature representations.
        :param num_classes: the number of supervised classes.
        """

        super(SupervisedModel, self).__init__()

        self._num_embeddings = num_embeddings
        self._embedding_dim = embedding_dim

        self.f = torch.nn.Sequential(
            OrderedDict(
                [
                    (
                        "embeddings",
                        torch.nn.EmbeddingBag(
                            num_embeddings=num_embeddings,
                            embedding_dim=embedding_dim,
                            sparse=False,
                        ),
                    ),
                    ("fc", torch.nn.Linear(embedding_dim, num_classes, bias=False)),
                ]
            )
        )

    def forward(self, inputs: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        return self.f.fc(self.f.embeddings(inputs, offsets))
