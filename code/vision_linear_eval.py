import os
from pathlib import Path
from typing import Tuple

import hydra
import numpy as np
import torch
import torch.optim as optim
import torchvision
import wandb
import yaml
from omegaconf import OmegaConf
from src.datasets.cifar import get_train_val_test_datasets
from src.datasets.utils import ClassDownSampler
from src.ddp import cleanup, init_ddp
from src.models.downstream import LinearClassifier
from src.models.vision.contrastive import ContrastiveModel
from src.transforms.simclr_transforms import create_simclr_data_augmentation
from src.utils.logger import get_logger
from src.utils.lr import calculate_lr_list
from src.utils.wandb import flatten_omegaconf
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


def evaluation(
    feature_encoder: ContrastiveModel,
    classifier,
    local_rank: int,
    data_loader: DataLoader,
    top_k: int = 5,
    normalize: bool = False,
) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    """
    Calculate loss and accuracy without parameter updates.

    :param feature_encoder: Instance of `ContrastiveModel`.
    :param classifier: Instance of `LinearClassifier`.
    :param local_rank: local rank.
    :param data_loader: validation data loader.
    :param top_k: k value of top_k accuracy.
    :param normalize: whether or not to normalize the feature representations.

    :return: Tuple of tensor that contains loss and number of correct samples, number of correct samples for top-k
        accuracy.
    """

    feature_encoder.eval()
    classifier.eval()

    sum_loss = torch.tensor([0.0]).to(local_rank)
    num_corrects = torch.tensor([0.0]).to(local_rank)
    num_corrects_top_k = torch.tensor([0.0]).to(local_rank)

    with torch.no_grad():
        for xs, ys in data_loader:
            ys = ys.to(local_rank)  # (B,)

            features = feature_encoder(xs.to(local_rank))  # (B, num_features)

            if normalize:
                features = torch.nn.functional.normalize(features, p=2, dim=1)

            pre_softmax = classifier(features)

            loss = torch.nn.functional.cross_entropy(pre_softmax, ys, reduction="sum")
            sum_loss += loss.item()

            pred_top_k = torch.topk(pre_softmax, dim=1, k=top_k)[1]
            pred_top_1 = pred_top_k[:, 0]

            num_corrects += pred_top_1.eq(ys.view_as(pred_top_1)).sum().item()
            num_corrects_top_k += (pred_top_k == ys.view(len(ys), 1)).sum().item()

    return sum_loss, num_corrects, num_corrects_top_k


@hydra.main(config_path="conf", config_name="vision_linear_eval")
def main(cfg: OmegaConf):
    logger = get_logger()

    use_cuda = torch.cuda.is_available()
    local_rank = int(os.environ["LOCAL_RANK"]) if use_cuda else "cpu"

    seed = cfg["seed"]
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    rnd = np.random.RandomState(seed)

    logger.info("local_rank:{}".format(local_rank))

    init_ddp(
        cfg=cfg, local_rank=local_rank, world_size=cfg["distributed"]["world_size"]
    )

    batch_size = cfg["optimizer"]["mini_batch_size"]
    weights_path = Path(cfg["target_weight_file"])

    contrastive_config_path = weights_path.parent / ".hydra" / "config.yaml"
    with open(contrastive_config_path) as f:
        contrastive_cfg = yaml.load(f, Loader=yaml.FullLoader)

    dataset_name = contrastive_cfg["dataset"]["name"]

    train_dataset, val_dataset, test_dataset = get_train_val_test_datasets(
        rnd=rnd,
        validation_ratio=contrastive_cfg["dataset"]["validation_ratio"],
        dataset_name=dataset_name,
    )
    down_sampler = ClassDownSampler(
        rnd=rnd, sklearn_train_size=contrastive_cfg["dataset"]["num_used_classes"]
    )
    train_dataset = down_sampler.fit_transform(train_dataset)
    val_dataset = down_sampler.transform(val_dataset)
    test_dataset = down_sampler.transform(test_dataset)

    if contrastive_cfg["dataset"]["simclr_data_augmentation"]:
        train_dataset.transform = create_simclr_data_augmentation(
            contrastive_cfg["dataset"]["strength"],
            size=contrastive_cfg["dataset"]["size"],
        )

        val_transform = torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor(),]
        )
        val_dataset.transform = val_transform
        test_dataset.transform = val_transform

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=DistributedSampler(
            train_dataset, rank=local_rank, shuffle=True, seed=seed, drop_last=True
        ),
        drop_last=True,
        pin_memory=True,
        num_workers=3,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=DistributedSampler(
            val_dataset, rank=local_rank, shuffle=False, seed=seed, drop_last=False
        ),
        pin_memory=True,
        num_workers=3,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        sampler=DistributedSampler(
            test_dataset, rank=local_rank, shuffle=False, seed=seed, drop_last=False
        ),
        pin_memory=True,
        num_workers=3,
    )

    num_classes = len(np.unique(train_dataset.targets))
    num_train_samples = len(train_dataset)
    num_val_samples = len(val_dataset)
    num_test_samples = len(test_dataset)

    if local_rank == 0:
        logger.info("Evaluation by using {}".format(weights_path.name))
        logger.info(
            "#train: {} #validation: {}, #test: {}, #classes: {}\n".format(
                num_train_samples, num_val_samples, num_test_samples, num_classes
            )
        )

        config = flatten_omegaconf(cfg)
        config["dataset.num_used_classes"] = contrastive_cfg["dataset"][
            "num_used_classes"
        ]
        wandb.init(
            dir=hydra.utils.get_original_cwd(),
            project="CURL",
            entity="INPUT_YOUR_ENTITY",
            config=config,
            tags=[dataset_name, "linear-eval"],
            group="seed-{}".format(seed),
        )

    # load pre-trained model
    feature_encoder = ContrastiveModel(
        base_cnn=contrastive_cfg["architecture"]["name"],
        d=contrastive_cfg["architecture"]["d"],
        is_cifar="cifar" in dataset_name,
    ).to(local_rank)

    feature_encoder = torch.nn.SyncBatchNorm.convert_sync_batchnorm(feature_encoder)
    feature_encoder = feature_encoder.to(local_rank)

    map_location = {"cuda:%d" % 0: "cuda:%d" % local_rank}
    state_dict = torch.load(weights_path, map_location=map_location)
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    # load weights trained on self-supervised task
    feature_encoder.load_state_dict(state_dict)

    # get the dimensionality of the representation
    if cfg["use_projection_head"]:
        num_last_units = feature_encoder.g.projection_head.linear2.out_features
    else:
        num_last_units = feature_encoder.g.projection_head.linear1.in_features
        feature_encoder.g = torch.nn.Identity()

    classifier = LinearClassifier(num_last_units, num_classes).to(local_rank)
    classifier = torch.nn.parallel.DistributedDataParallel(
        classifier, device_ids=[local_rank]
    )

    if cfg["optimizer"]["name"] == "sgd":
        optimizer = optim.SGD(
            params=classifier.parameters(),
            lr=cfg["optimizer"]["lr"],
            momentum=cfg["optimizer"]["momentum"],
            nesterov=cfg["optimizer"]["nesterov"],
            weight_decay=cfg["optimizer"]["weight_decay"],
        )

    if cfg["lr_scheduler"]["name"] == "cosine":
        num_iters = len(train_loader)  # because the drop=True
        lr_list = calculate_lr_list(cfg, batch_size=batch_size, num_iters=num_iters)
    else:
        ValueError("Unsupported learning rate scheduler")

    highest_val_acc = 0.0
    top_k = 5
    normalize: bool = cfg["normalize"]
    epochs = cfg["epochs"]
    num_train_samples_per_epoch = (
        batch_size * len(train_loader) * cfg["distributed"]["world_size"]
    )
    save_fname = Path(os.getcwd()) / cfg["output_model_name"]

    for epoch in range(epochs):
        classifier.train()
        feature_encoder.eval()
        train_loader.sampler.set_epoch(epoch)

        train_loss = torch.tensor([0.0]).to(local_rank)

        for batch_idx, (xs, ys) in enumerate(train_loader):

            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_list[epoch * num_iters + batch_idx]

            # instead of optimizer.zero_grad()
            # ref: https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#use-parameter-grad-none-instead-of-model-zero-grad-or-optimizer-zero-grad
            for param in classifier.parameters():
                param.grad = None

            with torch.no_grad():
                features = feature_encoder(xs.to(local_rank))  # (B, num_features)
                if normalize:
                    features = torch.nn.functional.normalize(features, p=2, dim=1)

            pre_softmax = classifier(features)  # (B, num_classes)

            loss = torch.nn.functional.cross_entropy(
                pre_softmax, ys.to(local_rank), reduction="mean"
            )
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch_size

        val_loss, val_num_corrects, val_num_corrects_top_k = evaluation(
            feature_encoder, classifier, local_rank, val_loader, top_k, normalize
        )

        # aggregate loss and accuracy over the world
        torch.distributed.reduce(train_loss, dst=0)
        torch.distributed.reduce(val_loss, dst=0)
        torch.distributed.reduce(val_num_corrects, dst=0)
        torch.distributed.reduce(val_num_corrects_top_k, dst=0)

        if local_rank == 0:
            # logging at end of each epoch

            # since `drop_last=True`
            train_loss = train_loss.item() / num_train_samples_per_epoch

            val_loss = val_loss.item() / num_val_samples
            val_acc = val_num_corrects.item() / num_val_samples * 100.0
            val_acc_top_k = val_num_corrects_top_k.item() / num_val_samples * 100.0

            logger.info(
                "epoch: {}/{} train loss: {:.4f} valid. loss: {:.4f}, valid. acc.: {:.1f} %, lr: {:.4f}".format(
                    epoch + 1,
                    cfg["epochs"],
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
                    "supervised_val_top-{}_acc".format(top_k): val_acc_top_k,
                },
                step=epoch,
            )

            if highest_val_acc <= val_acc:
                torch.save(classifier.state_dict(), save_fname)
                highest_val_acc = val_acc

    # eval on test dataset
    # without the following line, when rank zero saving the weights file,
    # the other ranks access the incomplete file
    torch.distributed.barrier()
    map_location = {"cuda:%d" % 0: "cuda:%d" % local_rank}
    classifier.load_state_dict(torch.load(save_fname, map_location=map_location))
    test_loss, test_num_corrects, test_num_corrects_top_k = evaluation(
        feature_encoder, classifier, local_rank, test_loader, top_k, normalize
    )

    torch.distributed.reduce(test_loss, dst=0)
    torch.distributed.reduce(test_num_corrects, dst=0)
    torch.distributed.reduce(test_num_corrects_top_k, dst=0)

    if local_rank == 0:
        test_loss = test_loss.item() / num_test_samples
        test_acc = test_num_corrects.item() / num_test_samples * 100.0
        test_top_k_acc = test_num_corrects_top_k.item() / num_test_samples * 100.0

        wandb.run.summary["supervised_test_loss"] = test_loss
        wandb.run.summary["supervised_test_acc"] = test_acc
        wandb.run.summary["supervised_test_top-{}_acc".format(top_k)] = test_top_k_acc
        wandb.save(str(save_fname))

    torch.distributed.barrier()
    cleanup()


if __name__ == "__main__":
    main()
