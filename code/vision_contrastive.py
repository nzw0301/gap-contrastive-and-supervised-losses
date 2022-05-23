import os
from pathlib import Path
from typing import Any, Dict, Tuple

import hydra
import numpy as np
import torch
import torch.optim as optim
from apex.parallel.LARC import LARC
from omegaconf import OmegaConf
from src.datasets.cifar import get_train_val_test_datasets
from src.datasets.contrastive import ContrastiveDataset
from src.datasets.utils import ClassDownSampler
from src.ddp import cleanup, init_ddp
from src.losses.contrastive_loss import ContrastiveLoss
from src.models.vision.contrastive import ContrastiveModel
from src.transforms.simclr_transforms import create_simclr_data_augmentation
from src.utils.logger import get_logger
from src.utils.lr import calculate_initial_lr, calculate_lr_list
from src.utils.wandb import flatten_omegaconf
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


def exclude_from_wt_decay(
    named_params, weight_decay: float, skip_list=("bias", "bn")
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    :param named_params: Model's named_params.
    :param weight_decay: weight_decay's parameter.
    :param skip_list: list of names to exclude weight decay.

    :return: dictionaries of params
    """
    # https://github.com/google-research/simclr/blob/3fb622131d1b6dee76d0d5f6aac67db84dab3800/model_util.py#L99

    params = []
    excluded_params = []

    for name, param in named_params:

        if not param.requires_grad:
            continue
        elif any(layer_name in name for layer_name in skip_list):
            excluded_params.append(param)
        else:
            params.append(param)

    return (
        {"params": params, "weight_decay": weight_decay},
        {"params": excluded_params, "weight_decay": 0.0},
    )


@hydra.main(config_path="conf", config_name="vision_contrastive")
def main(cfg: OmegaConf) -> None:
    logger = get_logger()

    use_cuda = torch.cuda.is_available()
    local_rank = int(os.environ["LOCAL_RANK"]) if use_cuda else "cpu"

    seed = cfg["seed"]
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(
        seed
    )  # note that this draws the same indices in contrastive_loss among world, but the data are different
    rnd = np.random.RandomState(seed)

    logger.info("local_rank:{}".format(local_rank))

    init_ddp(
        cfg=cfg, local_rank=local_rank, world_size=cfg["distributed"]["world_size"]
    )

    dataset_name = cfg["dataset"]["name"]
    batch_size = cfg["optimizer"]["mini_batch_size"]

    # do not use validation dataset
    train_dataset, _, _ = get_train_val_test_datasets(
        rnd=rnd,
        validation_ratio=cfg["dataset"]["validation_ratio"],
        dataset_name=dataset_name,
    )
    down_sampler = ClassDownSampler(
        rnd=rnd, sklearn_train_size=cfg["dataset"]["num_used_classes"]
    )
    train_dataset = down_sampler.fit_transform(train_dataset)

    train_dataset = ContrastiveDataset(
        train_dataset,
        rnd=rnd,
        num_n_pairs_per_sample=cfg["dataset"]["train_num_n_pairs_per_sample"],
        transform=train_dataset.transform
        if hasattr(train_dataset, "transform")
        else None,
        is_image_dataset=True,
    )

    if cfg["dataset"]["simclr_data_augmentation"]:
        train_dataset.transform = create_simclr_data_augmentation(
            cfg["dataset"]["strength"], size=cfg["dataset"]["size"]
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=DistributedSampler(
            train_dataset, rank=local_rank, shuffle=True, seed=seed, drop_last=True
        ),
        drop_last=True,
        num_workers=3,
        pin_memory=True,
    )

    num_train_samples = len(train_dataset)

    if local_rank == 0:
        import wandb

        logger.info("#train: {}\n".format(num_train_samples))

        wandb.init(
            dir=hydra.utils.get_original_cwd(),
            project="CURL",
            entity="INPUT_YOUR_ENTITY",
            config=flatten_omegaconf(cfg),
            tags=[dataset_name, "contrastive"],
            group="seed-{}".format(seed),
        )

    is_cifar = "cifar" in cfg["dataset"]["name"]

    model = ContrastiveModel(
        base_cnn=cfg["architecture"]["name"],
        d=cfg["architecture"]["d"],
        is_cifar=is_cifar,
    )
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.to(local_rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    contrastive_loss = ContrastiveLoss(
        local_rank,
        num_negatives=cfg["loss"]["neg_size"],
        normalize=cfg["loss"]["normalize"],
        reduction="mean",
    )

    if cfg["optimizer"]["name"] == "sgd":
        optimizer = LARC(
            optimizer=optim.SGD(
                params=exclude_from_wt_decay(
                    model.named_parameters(),
                    weight_decay=cfg["optimizer"]["weight_decay"],
                ),
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
    lowest_train_loss = np.inf
    epochs = cfg["epochs"]
    num_train_samples_per_epoch = (
        batch_size * num_iters * cfg["distributed"]["world_size"]
    )
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(epochs):
        model.train()
        train_loader.sampler.set_epoch(epoch)

        train_loss = torch.tensor([0.0]).to(local_rank)

        for batch_idx, (anchors, positives) in enumerate(train_loader):

            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_list[epoch * num_iters + batch_idx]

            # instead of optimizer.zero_grad()
            # ref: https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#use-parameter-grad-none-instead-of-model-zero-grad-or-optimizer-zero-grad
            for param in model.parameters():
                param.grad = None

            with torch.cuda.amp.autocast():
                anchors_features = model(anchors.to(local_rank))  # (B, num_features)
                positives_features = model(
                    positives.to(local_rank)
                )  # (B, num_features)

                loss = contrastive_loss(anchors_features, positives_features)  # (,)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.detach() * batch_size

        # aggregate loss and accuracy over the world
        torch.distributed.reduce(train_loss, dst=0)

        if local_rank == 0:

            # logging at end of each epoch
            train_loss = train_loss.item() / num_train_samples_per_epoch

            logger.info(
                "epoch: {}/{} train loss: {:.4f}, lr: {:.4f}".format(
                    epoch + 1, epochs, train_loss, optimizer.param_groups[0]["lr"]
                )
            )

            wandb.log(data={"contrastive_train_loss": train_loss}, step=epoch)

            if train_loss <= lowest_train_loss:
                lowest_train_loss = train_loss
                # save the model that minimizes the contrastive loss
                if epochs - epoch < (epochs * 0.15):
                    torch.save(model.state_dict(), save_fname)

            # every 200 epoch, save model
            if (epoch + 1) % 200 == 0:
                postfix = "_{}.pt".format(epoch + 1)
                torch.save(model.state_dict(), str(save_fname).replace(".pt", postfix))

    # store the best model
    if local_rank == 0:
        wandb.save(str(save_fname))

    # without waiting, "unhandled cuda error, NCCL" happens
    torch.distributed.barrier()
    # without this cleanup, some process might be alive for the next program
    cleanup()


if __name__ == "__main__":
    main()
