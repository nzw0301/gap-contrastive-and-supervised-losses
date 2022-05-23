from typing import Tuple

import torch
from torch.utils.data.dataloader import DataLoader

from ..models.downstream import MeanClassifier


def mean_classifier_eval(
    data_loader: DataLoader,
    device: torch.device,
    classifier: MeanClassifier,
    top_k: int = 5,
) -> Tuple[float, float, float]:
    """
    :param data_loader: DataLoader of downstream task.
    :param device: PyTorch's device instance.
    :param classifier: Instance of MeanClassifier.
    :param top_k: The number of top-k to calculate accuracy.

    :return: Tuple of top-1 accuracy, top-k accuracy, and mean classifier's cross-entropy loss.
    """

    assert not data_loader.drop_last

    num_samples = len(data_loader.dataset)
    top_1_correct = 0
    top_k_correct = 0
    loss = 0.0

    classifier.eval()
    with torch.no_grad():
        for xs, ys in data_loader:
            ys = ys.to(device)

            unnormalized_score = classifier(xs.to(device))
            pred_top_k = torch.topk(unnormalized_score, dim=1, k=top_k)[1]
            pred_top_1 = pred_top_k[:, 0]

            top_1_correct += pred_top_1.eq(ys.view_as(pred_top_1)).sum().item()
            if top_k > 1:
                top_k_correct += (pred_top_k == ys.view(len(ys), 1)).sum().item()
            else:
                top_k_correct = top_1_correct

            loss += torch.nn.functional.cross_entropy(
                unnormalized_score, ys, reduction="sum"
            ).item()

    return top_1_correct / num_samples, top_k_correct / num_samples, loss / num_samples
