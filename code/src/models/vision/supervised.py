import torch
from torchvision.models import resnet18, resnet50


class SupervisedModel(torch.nn.Module):
    def __init__(
        self, base_cnn: str = "resnet18", num_classes: int = 10, is_cifar: bool = True
    ) -> None:
        """
        :param base_cnn: name of backbone model.
        :param num_classes: the number of supervised classes.
        :param is_cifar: Whether CIFAR10/100 or not.
        """

        assert base_cnn in {"resnet18", "resnet50"}
        super(SupervisedModel, self).__init__()

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
                "`base_cnn` must be either `resnet18` or `resnet50`. `{}` is unsupported.".format(
                    base_cnn
                )
            )

        self.f.fc = torch.nn.Linear(num_last_hidden_units, num_classes, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:

        return self.f(inputs)
