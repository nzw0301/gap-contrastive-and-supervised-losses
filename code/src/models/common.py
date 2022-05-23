from collections import OrderedDict

import torch


class ProjectionHead(torch.nn.Module):
    def __init__(self, num_last_hidden_units: int, d: int) -> None:
        """
        The non-linear projection head for contrastive model.

        :param num_last_hidden_units: the dimensionality of the encoder's output representation.
        :param d: the dimensionality of output.
        """
        super(ProjectionHead, self).__init__()

        self.projection_head = torch.nn.Sequential(
            OrderedDict(
                [
                    (
                        "linear1",
                        torch.nn.Linear(
                            num_last_hidden_units, num_last_hidden_units, bias=False
                        ),
                    ),
                    ("bn1", torch.nn.BatchNorm1d(num_last_hidden_units)),
                    ("relu1", torch.nn.ReLU()),
                    ("linear2", torch.nn.Linear(num_last_hidden_units, d, bias=False)),
                ]
            )
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.projection_head(inputs)
