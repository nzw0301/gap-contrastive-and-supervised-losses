from pathlib import Path

import hydra
import numpy as np
import torch
import wandb
import yaml
from omegaconf import OmegaConf
from src.datasets.contrastive import ContrastiveDataset
from src.datasets.wiki3029 import collate_contrastive_batch, get_train_val_test_datasets
from src.losses.contrastive_loss import ContrastiveLoss
from src.models.language.contrastive import ContrastiveModel
from src.utils.logger import get_logger
from src.utils.wandb import flatten_omegaconf
from torch.utils.data import DataLoader


@hydra.main(config_path="conf", config_name="language_contrastive_eval")
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

    _, val_dataset, test_dataset = get_train_val_test_datasets(
        rnd=rnd,
        validation_ratio=contrastive_cfg["dataset"]["validation_ratio"],
        test_ratio=contrastive_cfg["dataset"]["test_ratio"],
        min_freq=contrastive_cfg["dataset"]["min_freq"],
        root=Path.home() / "pytorch_datasets",
        train_size=contrastive_cfg["dataset"]["num_used_classes"],
    )

    vocab_size = val_dataset.vocab_size
    batch_size = cfg["mini_batch_size"]
    num_val_samples = len(val_dataset)
    num_test_samples = len(test_dataset)
    num_classes = len(np.unique(val_dataset.targets))

    logger.info("device:{}".format(device))
    logger.info(
        f"#validation: {num_val_samples}, #test: {num_test_samples} #vocab: {vocab_size}, #clases: {num_classes}"
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

    # always normalize=True for theory
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
        for _ in range(cfg["num_loops"]):
            # positive pair is created by the following line, to reduce the randomness,
            # we evaluate contrastive loss `num_loops` times.
            contrastive_dataset = ContrastiveDataset(
                dataset,
                rnd=rnd,
                num_n_pairs_per_sample=1,
                transform=dataset.transform if hasattr(dataset, "transform") else None,
                is_image_dataset=False,
            )

            data_loader = DataLoader(
                contrastive_dataset,
                batch_size=batch_size,
                num_workers=9,
                pin_memory=True,
                collate_fn=collate_contrastive_batch,
                drop_last=True,
            )

            sum_loss = 0.0
            # use inference mode if we can use pytorch>=1.9.0
            with torch.no_grad():
                for (
                    batch_idx,
                    (anchors, anchor_offsets, positives, positives_offsets),
                ) in enumerate(data_loader):
                    anchors_features = feature_encoder(
                        anchors.to(device), anchor_offsets.to(device)
                    )  # (B, num_features)
                    positives_features = feature_encoder(
                        positives.to(device), positives_offsets.to(device)
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
