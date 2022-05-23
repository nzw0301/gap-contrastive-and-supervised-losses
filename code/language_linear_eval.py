import os
from pathlib import Path
from typing import Tuple

import hydra
import numpy as np
import torch
import wandb
import yaml
from omegaconf import OmegaConf
from src.datasets.wiki3029 import collate_supervised_batch, get_train_val_test_datasets
from src.models.downstream import LinearClassifier
from src.models.language.contrastive import ContrastiveModel
from src.utils.logger import get_logger
from src.utils.lr import calculate_lr_list
from src.utils.wandb import flatten_omegaconf
from torch.utils.data import DataLoader


def evaluation(
    feature_encoder: ContrastiveModel,
    classifier,
    device: torch.device,
    data_loader: DataLoader,
    top_k: int = 5,
    normalize: bool = False,
) -> Tuple[float, float, float]:
    """
    Calculate loss and accuracy without parameter updates.

    :param feature_encoder: Instance of `ContrastiveModel`.
    :param classifier: Instance of `LinearClassifier`.
    :param device: PyTorch's device instance.
    :param data_loader: validation data loader.
    :param top_k: k value of top_k accuracy.
    :param normalize: whether or not to normalize the feature representations.

    :return: Tuple of tensor that contains loss and number of correct samples, number of correct samples for top-k
        accuracy.
    """
    assert not data_loader.drop_last

    feature_encoder.eval()
    classifier.eval()

    num_samples = len(data_loader.dataset)
    sum_loss = 0.0
    num_corrects = 0.0
    num_corrects_top_k = 0.0

    with torch.no_grad():
        for batch_idx, (xs, ys, offsets) in enumerate(data_loader):
            ys = ys.to(device)
            features = feature_encoder(
                xs.to(device), offsets.to(device)
            )  # (batches, num_features)

            if normalize:
                features = torch.nn.functional.normalize(features, p=2, dim=1)

            pre_softmax = classifier(features)

            loss = torch.nn.functional.cross_entropy(pre_softmax, ys, reduction="sum")
            sum_loss += loss.item()

            pred_top_k = torch.topk(pre_softmax, dim=1, k=top_k)[1]
            pred_top_1 = pred_top_k[:, 0]

            num_corrects += pred_top_1.eq(ys.view_as(pred_top_1)).sum().item()
            num_corrects_top_k += (pred_top_k == ys.view(len(ys), 1)).sum().item()

    return (
        sum_loss / num_samples,
        num_corrects / num_samples,
        num_corrects_top_k / num_samples,
    )


@hydra.main(config_path="conf", config_name="language_linear_eval")
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

    batch_size = cfg["optimizer"]["mini_batch_size"]
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

    num_workers = 3
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=True,
        collate_fn=collate_supervised_batch,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_supervised_batch,
    )

    test_loader = DataLoader(
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

    # load weights trained on contrastive task
    feature_encoder.load_state_dict(state_dict)

    # get the dimensionality of the representation
    if cfg["use_projection_head"]:
        num_last_units = contrastive_cfg["architecture"]["embedding_dim"]
    else:
        num_last_units = contrastive_cfg["architecture"]["d"]
        feature_encoder.g = torch.nn.Identity()

    classifier = LinearClassifier(num_last_units, num_classes).to(device)

    if cfg["optimizer"]["name"] == "sgd":
        optimizer = torch.optim.SGD(
            params=classifier.parameters(),
            lr=cfg["optimizer"]["lr"],
            momentum=cfg["optimizer"]["momentum"],
            nesterov=cfg["optimizer"]["nesterov"],
            weight_decay=cfg["optimizer"]["weight_decay"],
        )
    else:
        NotImplementedError("Unsupported optimizer")

    if cfg["lr_scheduler"]["name"] == "cosine":
        num_iters = len(train_loader)  # because the drop=True
        lr_list = calculate_lr_list(cfg, batch_size=batch_size, num_iters=num_iters)
    else:
        NotImplementedError("Unsupported learning rate scheduler")

    config = flatten_omegaconf(cfg)
    config["dataset.num_used_classes"] = contrastive_cfg["dataset"]["num_used_classes"]

    wandb.init(
        dir=hydra.utils.get_original_cwd(),
        project="CURL",
        entity="INPUT_YOUR_ENTITY",
        config=config,
        tags=[contrastive_cfg["dataset"]["name"], "linear-eval"],
        group="seed-{}".format(seed),
    )

    highest_val_acc = 0.0
    top_k = 10  # same as Arora et al.
    normalize: bool = cfg["normalize"]
    epochs = cfg["epochs"]
    save_fname = Path(os.getcwd()) / cfg["output_model_name"]

    for epoch in range(epochs):
        classifier.train()
        feature_encoder.eval()

        train_loss = 0.0
        for batch_idx, (xs, ys, offsets) in enumerate(train_loader):

            # update learning rate
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_list[epoch * num_iters + batch_idx]

            optimizer.zero_grad()

            with torch.no_grad():
                features = feature_encoder(
                    xs.to(device), offsets.to(device)
                )  # (B, num_features)
                if normalize:
                    features = torch.nn.functional.normalize(features, p=2, dim=1)

            pre_softmax = classifier(features)  # (B, num_classes)

            loss = torch.nn.functional.cross_entropy(
                pre_softmax, ys.to(device), reduction="mean"
            )
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch_size

        val_loss, val_acc, val_top_k_acc = evaluation(
            feature_encoder, classifier, device, val_loader, top_k, normalize
        )

        train_loss = train_loss / (num_iters * batch_size)

        val_acc *= 100.0
        val_top_k_acc *= 100

        logger.info(
            "epoch: {}/{} train loss: {:.4f} valid. loss: {:.4f}, valid. acc.: {:.3f} %, lr: {:.4f}".format(
                epoch + 1,
                epochs,
                train_loss,
                val_loss,
                val_acc,
                optimizer.param_groups[0]["lr"],
            )
        )

        wandb.log(
            data={
                "supervised_train_loss": train_loss,
                "supervised_val_loss": val_loss,
                "supervised_val_acc": val_acc,
                "supervised_val_top-{}_acc".format(top_k): val_top_k_acc,
            },
            step=epoch,
        )

        if highest_val_acc <= val_acc:
            torch.save(classifier.state_dict(), save_fname)
            highest_val_acc = val_acc

    # eval test dataset
    classifier.load_state_dict(torch.load(save_fname, map_location=device))
    test_loss, test_acc, test_top_k_acc = evaluation(
        feature_encoder, classifier, device, test_loader, top_k
    )
    wandb.run.summary["supervised_test_loss"] = test_loss
    wandb.run.summary["supervised_test_acc"] = test_acc * 100.0
    wandb.run.summary["supervised_test_top-{}_acc".format(top_k)] = (
        test_top_k_acc * 100.0
    )
    wandb.save(str(save_fname))


if __name__ == "__main__":
    main()
