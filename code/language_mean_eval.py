import itertools
from pathlib import Path
from typing import Tuple

import hydra
import numpy as np
import torch
import wandb
import yaml
from omegaconf import OmegaConf
from src.datasets.downstream_dataset import DownstreamDataset
from src.datasets.wiki3029 import collate_supervised_batch, get_train_val_test_datasets
from src.models.downstream import MeanClassifier
from src.models.language.contrastive import ContrastiveModel
from src.utils.eval import mean_classifier_eval
from src.utils.logger import get_logger
from src.utils.wandb import flatten_omegaconf
from torch.utils.data import DataLoader


def convert_vectors(
    data_loader: torch.utils.data.DataLoader,
    model: ContrastiveModel,
    device: torch.device,
    normalized: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert experiment to feature representations.
    :param data_loader: Tata loader for raw experiment.
    :param model: Pre-trained model.
    :param device: PyTorch's device instance.
    :param normalized: Whether normalize the feature representation or not.

    :return: Tuple of tensors: features and labels.
    """

    new_X = []
    new_y = []
    model.eval()

    with torch.no_grad():
        for (xs, ys, offsets) in data_loader:
            fs = model(xs.to(device), offsets.to(device))

            if normalized:
                fs = torch.nn.functional.normalize(fs, p=2, dim=1)

            new_X.append(fs)
            new_y.append(ys)

    X = torch.cat(new_X).detach()
    y = torch.cat(new_y)

    return X, y


@hydra.main(config_path="conf", config_name="language_mean_eval")
def main(cfg: OmegaConf):
    logger = get_logger()

    use_cuda = torch.cuda.is_available()

    if use_cuda:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        gpu_id = (
            cfg["gpu_id"] % torch.cuda.device_count()
        )  # NOTE: GPU's id is one origin
        device = torch.device(f"cuda:{gpu_id}")
    else:
        device = torch.device("cpu")

    seed = cfg["seed"]
    torch.manual_seed(seed)
    rnd = np.random.RandomState(seed)

    weights_path = Path(cfg["target_weight_file"])
    logger.info("Evaluation by using {}".format(weights_path.name))

    contrastive_config_path = weights_path.parent / ".hydra" / "config.yaml"
    with open(contrastive_config_path) as f:
        contrastive_cfg = yaml.load(f, Loader=yaml.FullLoader)

    train_dataset, val_dataset, test_dataset = get_train_val_test_datasets(
        rnd=rnd,
        validation_ratio=contrastive_cfg["dataset"]["validation_ratio"],
        test_ratio=contrastive_cfg["dataset"]["test_ratio"],
        min_freq=contrastive_cfg["dataset"]["min_freq"],
        root=Path.home() / "pytorch_datasets",
        train_size=contrastive_cfg["dataset"]["num_used_classes"],
    )

    vocab_size = train_dataset.vocab_size
    batch_size = cfg["mini_batch_size"]

    num_workers = 3
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_supervised_batch,
        drop_last=True,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_supervised_batch,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_supervised_batch,
    )

    num_classes = len(np.unique(train_dataset.targets))
    num_train_samples = len(train_dataset)
    num_val_samples = len(val_dataset)
    num_test_samples = len(test_dataset)

    logger.info("device:{}".format(device))
    logger.info(
        f"#train: {num_train_samples}, #validation: {num_val_samples}, #test: {num_test_samples} #vocab: {vocab_size}, "
        f"#clases: {num_classes}"
    )

    feature_encoder = ContrastiveModel(
        num_embeddings=vocab_size,
        embedding_dim=contrastive_cfg["architecture"]["embedding_dim"],
        num_last_hidden_units=contrastive_cfg["architecture"]["d"],
        with_projection_head=contrastive_cfg["architecture"]["with_projection_head"],
    ).to(device)

    state_dict = torch.load(weights_path, map_location=device)

    # load weights trained on self-supervised task
    feature_encoder.load_state_dict(state_dict)

    # remove projection head
    if not cfg["use_projection_head"]:
        feature_encoder.g = torch.nn.Identity()

    normalize = cfg["normalize"]
    x, y = convert_vectors(
        train_dataloader, feature_encoder, device, normalized=normalize
    )
    downstream_train_dataset = DownstreamDataset(x, y)
    train_dataloader = DataLoader(downstream_train_dataset, batch_size=batch_size)

    classifier = MeanClassifier(
        weights=MeanClassifier.create_weights(
            downstream_train_dataset, num_classes=num_classes
        ).to(device)
    )

    x, y = convert_vectors(
        val_dataloader, feature_encoder, device, normalized=normalize
    )
    downstream_val_dataset = DownstreamDataset(x, y)
    val_dataloader = DataLoader(downstream_val_dataset, batch_size=batch_size)

    x, y = convert_vectors(
        test_dataloader, feature_encoder, device, normalized=normalize
    )
    downstream_test_dataset = DownstreamDataset(x, y)
    test_dataloader = DataLoader(downstream_test_dataset, batch_size=batch_size)

    config = flatten_omegaconf(cfg)
    config["dataset.num_used_classes"] = contrastive_cfg["dataset"]["num_used_classes"]

    wandb.init(
        dir=hydra.utils.get_original_cwd(),
        project="CURL",
        entity="INPUT_YOUR_ENTITY",
        config=config,
        tags=[contrastive_cfg["dataset"]["name"], "mean-eval"],
        group="seed-{}".format(seed),
    )

    top_k = 10  # same as Arora et al.
    data_names = ("train", "val", "test")
    metric_names = ("acc", "top-{}_acc".format(top_k), "loss")
    evaluation_results = {
        "supervised_{}_{}".format(*k): {}
        for k in itertools.product(data_names, metric_names)
    }

    # all mean vectors
    with torch.no_grad():

        for (data_name, data_loader) in zip(
            data_names, (train_dataloader, val_dataloader, test_dataloader)
        ):
            metrics = mean_classifier_eval(data_loader, device, classifier, top_k=top_k)
            for (metric_name, metric) in zip(metric_names, metrics):

                if "acc" in metric_name:
                    metric *= 100.0

                key = "supervised_{}_{}".format(data_name, metric_name)
                evaluation_results[key] = metric

    logger.info(
        "mean classifier's train acc: {:.3f}, valid. acc: {:.3f}".format(
            evaluation_results["supervised_train_acc"],
            evaluation_results["supervised_val_acc"],
        )
    )

    wandb.log(evaluation_results)


if __name__ == "__main__":
    main()
