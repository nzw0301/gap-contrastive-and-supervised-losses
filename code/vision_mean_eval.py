import itertools
from pathlib import Path
from typing import Tuple

import hydra
import numpy as np
import torch
import wandb
import yaml
from omegaconf import OmegaConf
from src.datasets.cifar import get_train_val_test_datasets
from src.datasets.downstream_dataset import DownstreamDataset
from src.datasets.utils import ClassDownSampler
from src.models.downstream import MeanClassifier
from src.models.vision.contrastive import ContrastiveModel
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
        for xs, ys in data_loader:
            fs = model(xs.to(device))
            if normalized:
                fs = torch.nn.functional.normalize(fs, p=2, dim=1)
            new_X.append(fs)
            new_y.append(ys)

    X = torch.cat(new_X).detach()
    y = torch.cat(new_y)

    return X, y


@hydra.main(config_path="conf", config_name="vision_mean_eval")
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
    contrastive_config_path = weights_path.parent / ".hydra" / "config.yaml"

    with open(contrastive_config_path) as f:
        contrastive_cfg = yaml.load(f, Loader=yaml.FullLoader)

    train_dataset, val_dataset, test_dataset = get_train_val_test_datasets(
        rnd=rnd,
        validation_ratio=contrastive_cfg["dataset"]["validation_ratio"],
        dataset_name=contrastive_cfg["dataset"]["name"],
    )
    down_sampler = ClassDownSampler(
        rnd=rnd, sklearn_train_size=contrastive_cfg["dataset"]["num_used_classes"]
    )
    train_dataset = down_sampler.fit_transform(train_dataset)
    val_dataset = down_sampler.transform(val_dataset)
    test_dataset = down_sampler.transform(test_dataset)

    num_train_samples = len(train_dataset)
    num_val_samples = len(val_dataset)
    num_test_samples = len(test_dataset)
    num_classes = len(np.unique(train_dataset.targets))
    batch_size = cfg["mini_batch_size"]
    num_workers = 3

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, num_workers=num_workers
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, num_workers=num_workers
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, num_workers=num_workers
    )

    logger.info(
        f"#train: {num_train_samples}, #validation: {num_val_samples}, #test: {num_test_samples},"
        f"#classes: {num_classes}"
    )

    feature_encoder = ContrastiveModel(
        base_cnn=contrastive_cfg["architecture"]["name"],
        d=contrastive_cfg["architecture"]["d"],
        is_cifar=True,
    ).to(device)

    feature_encoder = torch.nn.SyncBatchNorm.convert_sync_batchnorm(feature_encoder)
    feature_encoder = feature_encoder.to(device)

    config = flatten_omegaconf(cfg)
    config["dataset.num_used_classes"] = contrastive_cfg["dataset"]["num_used_classes"]

    top_k = 5
    data_names = ("train", "val", "test")
    metric_names = ("acc", "top-{}_acc".format(top_k), "loss")

    if config["eval_all_checkpoints"]:
        target_pt_paths = weights_path.parent.glob("*.pt")
    else:
        target_pt_paths = [weights_path]

    # evaluate all pytorch checkpoints
    for weights_path in target_pt_paths:
        logger.info("Evaluation by using {}".format(weights_path.name))

        # load weights trained on contrastive task
        state_dict = torch.load(weights_path, map_location=device)
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        feature_encoder.load_state_dict(state_dict)

        if not cfg["use_projection_head"]:
            feature_encoder.g = torch.nn.Identity()

        X, y = convert_vectors(
            train_dataloader, feature_encoder, device, normalized=cfg["normalize"]
        )
        downstream_train_dataset = DownstreamDataset(X, y)
        # since MeanClassifier does not have any trainable parameters, we can set shuffle=False
        downstream_train_dataloader = DataLoader(
            downstream_train_dataset, batch_size=batch_size
        )

        classifier = MeanClassifier(
            weights=MeanClassifier.create_weights(
                downstream_train_dataset, num_classes=num_classes
            ).to(device)
        )

        X, y = convert_vectors(
            val_dataloader, feature_encoder, device, normalized=cfg["normalize"]
        )
        downstream_val_dataset = DownstreamDataset(X, y)
        downstream_val_dataloader = DataLoader(
            downstream_val_dataset, batch_size=batch_size
        )

        X, y = convert_vectors(
            test_dataloader, feature_encoder, device, normalized=cfg["normalize"]
        )
        downstream_test_dataset = DownstreamDataset(X, y)
        downstream_test_dataloader = DataLoader(
            downstream_test_dataset, batch_size=batch_size
        )

        config["target_weight_file"] = str(weights_path)
        wandb_run = wandb.init(
            dir=hydra.utils.get_original_cwd(),
            project="CURL",
            entity="INPUT_YOUR_ENTITY",
            config=config,
            tags=[contrastive_cfg["dataset"]["name"], "mean-eval"],
            group="seed-{}".format(seed),
        )

        evaluation_results = {
            "supervised_{}_{}".format(*k): {}
            for k in itertools.product(data_names, metric_names)
        }

        # all mean vectors
        with torch.no_grad():
            for (data_name, data_loader) in zip(
                data_names,
                (
                    downstream_train_dataloader,
                    downstream_val_dataloader,
                    downstream_test_dataloader,
                ),
            ):

                metrics = mean_classifier_eval(
                    data_loader, device, classifier, top_k=top_k
                )

                for (metric_name, metric) in zip(metric_names, metrics):

                    if "acc" in metric_name:
                        metric *= 100.0

                    key = "supervised_{}_{}".format(data_name, metric_name)
                    evaluation_results[key] = metric

            logger.info(
                "mean classifier's accuracy: train: {:.3f}, valid.: {:.3f}, test: {:.3f}".format(
                    evaluation_results["supervised_train_acc"],
                    evaluation_results["supervised_val_acc"],
                    evaluation_results["supervised_test_acc"],
                )
            )

        wandb.log(evaluation_results)
        wandb_run.finish()


if __name__ == "__main__":
    main()
