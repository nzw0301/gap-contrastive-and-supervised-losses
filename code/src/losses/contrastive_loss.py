from typing import Union

import torch


class ContrastiveLoss(torch.nn.Module):
    def __init__(
        self,
        device: Union[torch.device, int],
        num_negatives: int,
        normalize: bool = True,
        replace: bool = False,
        reduction: str = "mean",
    ) -> None:
        """
        Contrastive loss class. This class supports softmax cross-entropy.

        :param replace: If True, negative samples are sampled with replacement, otherwise without replacement.
        :param reduction: Same as PyTorch's reduction parameter of loss functions.
        """

        if reduction not in {"none" "mean", "sum"}:
            ValueError(
                '`reduction` should be "none", "mean", or "sum". Not {}'.format(
                    reduction
                )
            )

        self._device = device
        self._K = num_negatives
        self._normalize = normalize
        self._replace = replace
        self._reduction = reduction

        super(ContrastiveLoss, self).__init__()

    def forward(
        self, anchor_features: torch.Tensor, positive_features: torch.Tensor
    ) -> torch.Tensor:
        """
        :param anchor_features: shape (mini-batch, num-features).
        :param positive_features: shape (mini-batch, num-features).
        :return: Tensor of loss.
        """

        B = len(anchor_features)
        K = min(B - 1, self._K)  # to deal with the end of iterations.

        if self._normalize:
            anchor_features = torch.nn.functional.normalize(anchor_features, p=2, dim=1)
            positive_features = torch.nn.functional.normalize(
                positive_features, p=2, dim=1
            )

        # taking dot-product
        sim01 = torch.matmul(
            anchor_features, positive_features.t()
        )  # diagonal is the positive
        sim00 = torch.matmul(positive_features, positive_features.t())

        # keep positive dot products
        positive_score = torch.diagonal(sim01).reshape(B, -1)  # (B, 1)

        # candidates of negative samples
        # https://discuss.pytorch.org/t/keep-off-diagonal-elements-only-from-square-matrix/54379/2
        candidates = torch.cat(
            [
                sim00.flatten()[1:].view(B - 1, B + 1)[:, :-1].reshape(B, B - 1),
                sim01.flatten()[1:].view(B - 1, B + 1)[:, :-1].reshape(B, B - 1),
            ],
            dim=1,
        )  # (B, 2B-2)

        # negative samples in the mini-batch
        if self._replace:
            mask = torch.topk(torch.rand(B, K, 2 * B - 2, device=self._device), k=1)[
                1
            ]  # generate random indices
            mask = mask.view(B, K)
        else:
            mask = torch.topk(torch.rand(B, B - 1, device=self._device), K)[
                1
            ]  # generate random indices

            # we have two representations per input sample, so we randomly select one representation without duplications
            # so randomly select one of view.
            mask[torch.rand(B, K, device=self._device) > 0.5] += B - 1

        # if pytorch >= 1.9.0; torch.take_along_dim(candidates, mask, dim=1)
        neg_scores = torch.gather(candidates, 1, mask)

        # merge positive / negative scores
        scores = torch.cat([positive_score, neg_scores], dim=1)  # (B, K+1)
        labels = torch.zeros(B, dtype=torch.long, device=self._device)

        return torch.nn.functional.cross_entropy(
            scores, labels, reduction=self._reduction
        )
