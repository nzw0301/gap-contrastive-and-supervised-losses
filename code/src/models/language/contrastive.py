import torch

from ..abs_contrastive import AbsContrastiveModel
from ..common import ProjectionHead


class ContrastiveModel(torch.nn.Module, AbsContrastiveModel):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int = 10,
        num_last_hidden_units: int = 128,
        with_projection_head=True,
    ) -> None:
        """
        :param num_embeddings: The size of vocabulary
        :param embedding_dim: the dimensionality of feature representations.
        :param num_last_hidden_units: the number of units in the final layer. If `with_projection_head` is False,
            this value is ignored.
        :param with_projection_head: bool flag whether or not to use additional linear layer whose dimensionality is
            `num_last_hidden_units`.
        """

        super(ContrastiveModel, self).__init__()

        self._num_embeddings = num_embeddings
        self._embedding_dim = embedding_dim
        self._num_last_hidden_units = num_last_hidden_units
        self._with_projection_head = with_projection_head

        self.f = torch.nn.EmbeddingBag(
            num_embeddings=num_embeddings, embedding_dim=embedding_dim, sparse=False
        )

        if self._with_projection_head:
            self.g = ProjectionHead(embedding_dim, num_last_hidden_units)
        else:
            self.g = torch.nn.Identity()

    def encode(self, inputs: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        """
        taken inputs and its offsets, extract feature representation.
        """
        return self.f(inputs, offsets)  # (B, embedding_dim)

    def forward(self, inputs: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        """
        taken inputs and its offsets, extract feature representation, then apply additional feature transform
        to calculate contrastive loss.
        """
        h = self.encode(inputs, offsets)
        z = self.g(h)
        return z  # (B, embedding_dim) or (B, num_last_hidden_units) depending on `with_projection_head`
