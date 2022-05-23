import os
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.optim as optim
import wandb
from apex.parallel.LARC import LARC
from omegaconf import OmegaConf
from src.datasets.contrastive import ContrastiveDataset
from src.datasets.wiki3029 import collate_contrastive_batch, get_train_val_test_datasets
from src.losses.contrastive_loss import ContrastiveLoss
from src.models.language.contrastive import ContrastiveModel
from src.utils.logger import get_logger
from src.utils.lr import calculate_initial_lr, calculate_lr_list
from src.utils.wandb import flatten_omegaconf
from torch.utils.data import DataLoader


@hydra.main(config_path="conf", config_name="language_contrastive")
def main(cfg: OmegaConf) -> None:
    logger = get_logger()

    use_cuda = torch.cuda.is_available()

    if use_cuda:
        gpu_id = (
            cfg["gpu_id"] % torch.cuda.device_count()
        )  # NOTE: GPU's id is one origin when we use gnu-parallel.
        device = torch.device(f"cuda:{gpu_id}")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        device = torch.device("cpu")

    seed = cfg["seed"]
    torch.manual_seed(seed)
    rnd = np.random.RandomState(seed)

    batch_size = cfg["optimizer"]["mini_batch_size"]

    # no use of validation and test datasets
    # class down sampling is performed in `convert_dataset_to_contrastive_dataset`.
    train_dataset, _, _ = get_train_val_test_datasets(
        rnd=rnd,
        validation_ratio=cfg["dataset"]["validation_ratio"],
        test_ratio=cfg["dataset"]["test_ratio"],
        min_freq=cfg["dataset"]["min_freq"],
        root=Path.home() / "pytorch_datasets",
        train_size=cfg["dataset"]["num_used_classes"],
    )
    vocab_size = train_dataset.vocab_size

    train_dataset = ContrastiveDataset(
        train_dataset,
        rnd=rnd,
        num_n_pairs_per_sample=cfg["dataset"]["train_num_n_pairs_per_sample"],
        transform=train_dataset.transform
        if hasattr(train_dataset, "transform")
        else None,
        is_image_dataset=False,
    )

    num_workers = 9
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_contrastive_batch,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=3,
    )

    num_train_samples = len(train_dataset)

    logger.info("device:{}".format(device))
    logger.info("#train: {} #vocab: {}\n".format(num_train_samples, vocab_size))

    wandb.init(
        dir=hydra.utils.get_original_cwd(),
        project="CURL",
        entity="INPUT_YOUR_ENTITY",
        config=flatten_omegaconf(cfg),
        tags=["wiki3029", "contrastive"],
        group="seed-{}".format(seed),
    )

    model = ContrastiveModel(
        num_embeddings=vocab_size,
        embedding_dim=cfg["architecture"]["embedding_dim"],
        num_last_hidden_units=cfg["architecture"]["d"],
        with_projection_head=cfg["architecture"]["with_projection_head"],
    ).to(device)

    contrastive_loss = ContrastiveLoss(
        device,
        num_negatives=cfg["loss"]["neg_size"],
        normalize=cfg["loss"]["normalize"],
        reduction="mean",
    )

    if cfg["optimizer"]["name"] == "sgd":
        optimizer = LARC(
            optimizer=optim.SGD(
                params=model.parameters(),
                lr=calculate_initial_lr(cfg, batch_size),
                momentum=cfg["optimizer"]["momentum"],
                nesterov=cfg["optimizer"]["nesterov"],
                weight_decay=cfg["optimizer"]["weight_decay"],
            ),
            trust_coefficient=0.001,
            clip=False,
        )
    else:
        raise ValueError("Unsupported optimizer")

    if cfg["lr_scheduler"]["name"] == "cosine":
        num_iters = len(train_loader)
        lr_list = calculate_lr_list(cfg, batch_size=batch_size, num_iters=num_iters)

    else:
        raise ValueError("Unsupported learning scheduler")

    save_fname = Path(os.getcwd()) / cfg["output_model_name"]
    epochs = cfg["epochs"]
    lowest_train_loss = np.inf

    for epoch in range(epochs):
        model.train()

        train_loss = 0.0
        for (
            batch_idx,
            (anchors, anchor_offsets, positives, positives_offsets),
        ) in enumerate(train_loader):

            # update learning rate
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_list[epoch * num_iters + batch_idx]

            # instead of optimizer.zero_grad()
            # ref: https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#use-parameter-grad-none-instead-of-model-zero-grad-or-optimizer-zero-grad
            for param in model.parameters():
                param.grad = None

            anchors_features = model(
                anchors.to(device), anchor_offsets.to(device)
            )  # (B, num_features)
            positives_features = model(
                positives.to(device), positives_offsets.to(device)
            )  # (B, num_features)

            loss = contrastive_loss(anchors_features, positives_features)  # (,)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch_size

        # logging at end of each epoch
        train_loss = train_loss / (num_iters * batch_size)

        logger.info(
            "epoch: {}/{} train loss: {:.4f} lr: {:.4f}".format(
                epoch + 1, epochs, train_loss, optimizer.param_groups[0]["lr"]
            )
        )

        wandb.log(data={"contrastive_train_loss": train_loss}, step=epoch)

        if train_loss <= lowest_train_loss:
            torch.save(model.state_dict(), save_fname)
            lowest_train_loss = train_loss

    # store the best model
    wandb.save(str(save_fname))


if __name__ == "__main__":
    main()
