from pathlib import Path

import hydra
import numpy as np
import torch
import wandb
import yaml
from omegaconf import OmegaConf
from src.datasets.cifar import get_train_val_test_datasets
from src.datasets.contrastive import ContrastiveDataset
from src.datasets.utils import ClassDownSampler
from src.losses.contrastive_loss import ContrastiveLoss
from src.models.vision.contrastive import ContrastiveModel
from src.utils.logger import get_logger
from src.utils.wandb import flatten_omegaconf
from torch.utils.data import DataLoader


@hydra.main(config_path="conf", config_name="vision_contrastive_eval")
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
        dataset_name=contrastive_cfg["dataset"]["name"],
    )
    down_sampler = ClassDownSampler(
        rnd=rnd, sklearn_train_size=contrastive_cfg["dataset"]["num_used_classes"]
    )
    down_sampler.fit(train_dataset)
    val_dataset = down_sampler.transform(val_dataset)
    test_dataset = down_sampler.transform(test_dataset)
    del train_dataset

    num_val_samples = len(val_dataset)
    num_test_samples = len(test_dataset)
    num_classes = len(np.unique(val_dataset.targets))

    batch_size = cfg["mini_batch_size"]

    logger.info(
        f"#validation: {num_val_samples}, #test: {num_test_samples}, #classes: {num_classes}"
    )

    feature_encoder = ContrastiveModel(
        base_cnn=contrastive_cfg["architecture"]["name"],
        d=contrastive_cfg["architecture"]["d"],
        is_cifar=True,
    ).to(device)

    feature_encoder = torch.nn.SyncBatchNorm.convert_sync_batchnorm(feature_encoder)
    feature_encoder = feature_encoder.to(device)

    # load weights trained on contrastive task
    state_dict = torch.load(weights_path, map_location=device)
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    feature_encoder.load_state_dict(state_dict)

    contrastive_loss = ContrastiveLoss(
        device,
        num_negatives=contrastive_cfg["loss"]["neg_size"],
        normalize=True,
        reduction="sum",
    )

    config = flatten_omegaconf(cfg)
    config["dataset.num_used_classes"] = contrastive_cfg["dataset"]["num_used_classes"]
    wandb.init(
        dir=hydra.utils.get_original_cwd(),
        project="CURL",
        entity="INPUT_YOUR_ENTITY",
        config=config,
        tags=[contrastive_cfg["dataset"]["name"], "contrastive-eval"],
        group="seed-{}".format(seed),
    )

    data_names = ("val", "test")
    feature_encoder.eval()

    for data_name, dataset in zip(data_names, (val_dataset, test_dataset)):
        contrastive_losses = []

        if data_name == "val":
            _batch_size = min(batch_size, num_val_samples)
        else:
            _batch_size = min(batch_size, num_test_samples)

        for _ in range(cfg["num_loops"]):
            contrastive_dataset = ContrastiveDataset(
                dataset,
                rnd=rnd,
                num_n_pairs_per_sample=1,
                transform=dataset.transform,
                is_image_dataset=False,
            )

            data_loader = DataLoader(
                contrastive_dataset,
                batch_size=_batch_size,
                num_workers=9,
                pin_memory=True,
                drop_last=True,
            )
            sum_loss = 0.0
            # use inference mode if we can use pytorch>=1.9.0
            with torch.no_grad():
                for batch_idx, (anchors, positives) in enumerate(data_loader):
                    anchors_features = feature_encoder(
                        anchors.to(device)
                    )  # (B, num_features)
                    positives_features = feature_encoder(
                        positives.to(device)
                    )  # (B, num_features)

                    loss = contrastive_loss(anchors_features, positives_features)  # (,)
                    sum_loss += loss.item()

            contrastive_losses.append(
                sum_loss / (len(data_loader) * data_loader.batch_size)
            )

        wandb.log(
            {
                "contrastive_{}_loss".format(data_name): np.mean(contrastive_losses),
                "contrastive_{}_loss_std".format(data_name): np.std(contrastive_losses),
            }
        )


if __name__ == "__main__":
    main()
