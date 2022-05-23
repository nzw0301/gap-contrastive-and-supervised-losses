from collections import OrderedDict

import torch
from torchvision.models import resnet18, resnet50

from ..abs_contrastive import AbsContrastiveModel
from ..common import ProjectionHead


class ContrastiveModel(torch.nn.Module, AbsContrastiveModel):
    def __init__(
        self, base_cnn: str = "resnet18", d: int = 128, is_cifar: bool = True
    ) -> None:
        """
        :param base_cnn: The backbone's model name. resnet18 or resnet50.
        :param d: The dimensionality of the output feature.
        :param is_cifar: model is for CIFAR10/100 or not. If true, the model is modified by following SimCLR.
        """

        assert base_cnn in {"resnet18", "resnet50"}
        super(ContrastiveModel, self).__init__()

        if base_cnn == "resnet50":
            self.f = resnet50()
            num_last_hidden_units = 2048
        elif base_cnn == "resnet18":
            self.f = resnet18()
            num_last_hidden_units = 512

            if is_cifar:
                # replace the first conv2d with smaller conv
                self.f.conv1 = torch.nn.Conv2d(
                    in_channels=3,
                    out_channels=64,
                    stride=(1, 1),
                    kernel_size=(3, 3),
                    padding=(3, 3),
                    bias=False,
                )

                # remove the first max pool
                self.f.maxpool = torch.nn.Identity()

        else:
            raise ValueError(
                "`base_cnn` must be either `resnet18`, `resnet50``. `{}` is unsupported.".format(
                    base_cnn
                )
            )

        # drop the last classification layer
        self.f.fc = torch.nn.Identity()

        # non-linear projection head
        self.g = ProjectionHead(num_last_hidden_units, d)

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Return features before projection head.
        :param inputs: FloatTensor that contains images.

        :return: feature representations.
        """

        return self.f(inputs)  # N x num_last_hidden_units

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:

        h = self.encode(inputs)
        z = self.g(h)
        return z  # N x d
