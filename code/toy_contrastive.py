import json
import os
import sys
from itertools import product

import hydra
import numpy as np
import torch
import wandb
from omegaconf import OmegaConf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from src.datasets.contrastive import ContrastiveDataset
from src.datasets.downstream_dataset import DownstreamDataset
from src.losses.contrastive_loss import ContrastiveLoss
from src.models.downstream import MeanClassifier
from src.utils.logger import get_logger
from src.utils.wandb import flatten_omegaconf
from torch.utils.data import DataLoader
from vision_mean_eval import convert_vectors, mean_classifier_eval


class ContrastiveModel(torch.nn.Module):
    def __init__(self, num_hidden=128, num_last_units=128, num_layers=1):
        super(ContrastiveModel, self).__init__()
        self.f = torch.nn.functional.relu
        self.h0 = torch.nn.Linear(2, num_hidden)
        self.hs = torch.nn.ModuleList(
            [torch.nn.Linear(num_hidden, num_hidden) for _ in range(num_layers - 1)]
        )
        self.hz = torch.nn.Linear(num_hidden, num_last_units, bias=False)

    def forward(self, x):
        y = self.f(self.h0(x))
        for i in range(len(self.hs)):
            y = self.f(self.hs[i](y))
        return self.hz(y)


def rotate(X, degree=180):
    """
    rotate 2d data using degree
    """
    theta = np.radians(degree)

    rotation_matrix = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]],
    )

    return X.dot(rotation_matrix)


def generate_data(dataset_name, num_latent_classes, num_samples_per_class, rnd):
    if dataset_name == "norm":
        x, y = _generate_data_norm(num_latent_classes, num_samples_per_class, rnd)

    elif dataset_name == "circle":
        x, y = _generate_data_circle(num_latent_classes, num_samples_per_class, rnd)

    else:
        raise ValueError("Unsupported dataset")

    return x, y


def _generate_data_norm(num_latent_classes, num_samples_per_class, rnd):
    """
    Generate toy training dataset for classification.
    Generate `num_samples_per_class` 2d samples,
    then `num_latent_classes` rotated samples and each ratation degree represents class label.
    """

    mean = [0, num_latent_classes + 1.0]
    loc = np.array(mean)
    xs = rnd.normal(loc, size=(num_samples_per_class, 2))

    X = []
    Y = []
    degrees = np.arange(num_latent_classes) * 360 / num_latent_classes
    for label, degree in enumerate(degrees):
        X.append(rotate(xs, degree))
        Y += [label] * num_samples_per_class

    return np.concatenate(X), Y


def _generate_data_circle(num_latent_classes, num_samples_per_class, rnd):
    """
    Generate toy training dataset for classification.
    Generate `num_samples_per_class` 2d samples with different class-wise norms.
    """

    X = []
    Y = []
    for label in range(num_latent_classes):
        xs = rnd.rand(num_samples_per_class, 2) - 0.5
        xs = normalize(xs) * (label + 1) / 2.0
        X.append(xs)
        Y += [label] * num_samples_per_class

    return np.concatenate(X), Y


def val_contrastive_eval(data_loader, neg_size, normalize, model, device, replace):
    sum_loss = 0.0
    contrastive_loss = ContrastiveLoss(
        device=device, num_negatives=neg_size, normalize=normalize, replace=replace
    )
    model.eval()

    with torch.no_grad():
        batch_size = None
        for anchors, positives in data_loader:
            anchors_features = model(anchors.to(device))
            positives_features = model(positives.to(device))
            loss = contrastive_loss.forward(anchors_features, positives_features)
            sum_loss += loss.item() * len(anchors)
            batch_size = len(anchors)

        sum_loss /= len(data_loader) * batch_size

    return sum_loss


# validation function
def validation(
    model,
    supervised_train_data_loader,
    supervised_validation_data_loader,
    validation_data_loader,
    neg_size,
    normalize,
    device,
    replace,
):
    num_workers = 0
    num_classes = len(np.unique(supervised_train_data_loader.dataset.targets))
    top_k = 1

    model.eval()

    val_cont_loss = val_contrastive_eval(
        validation_data_loader, neg_size, normalize, model, device, replace
    )

    # extract feature representation
    x, y = convert_vectors(
        supervised_train_data_loader, model, device, normalized=normalize
    )  # feature vectors

    downstream_training_dataset = DownstreamDataset(x, y)
    train_loader = DataLoader(
        downstream_training_dataset,
        batch_size=512,
        shuffle=False,
        num_workers=num_workers,
    )

    # create mean classifier
    classifier = MeanClassifier(
        weights=MeanClassifier.create_weights(
            downstream_training_dataset, num_classes=num_classes, normalize=normalize
        ).to(device)
    )

    x, y = convert_vectors(
        supervised_validation_data_loader, model, device, normalized=normalize
    )
    downstream_val_dataset = DownstreamDataset(x, y)
    val_dataloader = DataLoader(
        downstream_val_dataset, batch_size=512, shuffle=False, num_workers=num_workers
    )

    train_acc, _, train_sup_loss = mean_classifier_eval(
        train_loader, device, classifier, top_k=top_k
    )
    val_acc, _, val_sup_loss = mean_classifier_eval(
        val_dataloader, device, classifier, top_k=top_k
    )

    return (
        train_acc * 100.0,
        val_acc * 100.0,
        train_sup_loss,
        val_sup_loss,
        val_cont_loss,
    )


def save_results(cfg: OmegaConf, results: dict) -> None:
    with open(cfg["output_log_name"], "w") as f:
        json.dump(results, f)
        wandb.save(cfg["output_log_name"])

    sys.path.append(
        os.path.dirname(hydra.utils.to_absolute_path(__file__)) + "/../scripts"
    )
    from plot_toy_trajectory import plot_trajectory

    plot_trajectory(results, cfg["output_plot_name"])
    wandb.save(cfg["output_plot_name"])


def plot_representation(data_loader, model, device, epoch):
    import matplotlib.pyplot as plt

    plt.clf()
    x, y = convert_vectors(data_loader, model, device, normalized=True)
    x = x.to("cpu").detach().numpy().copy()
    y = y.to("cpu").detach().numpy().copy()

    for y_ in np.unique(y):
        plt.scatter(x[y == y_, 0], x[y == y_, 1])

    plt.title(f"epoch: {epoch}")
    plt.savefig(f"plots/{epoch}.pdf")


@hydra.main(config_path="conf", config_name="toy_contrastive")
def main(cfg: OmegaConf):
    logger = get_logger()

    seed = cfg["seed"]
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    rnd = np.random.RandomState(seed)

    dataset_name = cfg["dataset"]["name"]
    batch_size = cfg["optimizer"]["mini_batch_size"]
    training_ratio = cfg["dataset"]["training_ratio"]
    num_latent_classes = cfg["dataset"]["num_latent_classes"]  # C
    neg_samples = cfg["loss"]["neg_size"]  # K
    num_samples_per_class = cfg["dataset"]["num_samples_per_class"]
    num_n_pairs_per_sample = cfg["dataset"]["num_n_pairs_per_sample"]
    epochs = cfg["epochs"]

    if torch.cuda.is_available():
        gpu_id = cfg["gpu_id"] % torch.cuda.device_count()
        device = torch.device(f"cuda:{gpu_id}")
    else:
        device = torch.device("cpu")

    results = {}
    for num_latent_classes, neg_size in product(
        range(num_latent_classes, num_latent_classes + 1),
        range(neg_samples, neg_samples + 1),
    ):
        wandb.init(
            dir=hydra.utils.get_original_cwd(),
            project="CURL",
            entity="INPUT_YOUR_ENTITY",
            config=flatten_omegaconf(cfg),
            tags=[dataset_name, "contrastive"],
            mode="online",
        )

        if num_latent_classes not in results:
            results[num_latent_classes] = {}
        if neg_size not in results[num_latent_classes]:
            results[num_latent_classes][neg_size] = {
                "contrastive_loss": [],
                "train_acc_history": [],
                "val_acc_history": [],
                "train_sup_loss": [],
                "val_sup_loss": [],
                "val_cont_loss": [],
            }

        X, y = generate_data(
            dataset_name, num_latent_classes, num_samples_per_class, rnd
        )

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, train_size=training_ratio, stratify=y, random_state=rnd
        )

        supervised_train_dataset = DownstreamDataset(
            torch.FloatTensor(X_train), y_train
        )
        supervised_val_dataset = DownstreamDataset(torch.FloatTensor(X_val), y_val)

        contrastive_dataset = ContrastiveDataset(
            supervised_train_dataset,
            rnd=rnd,
            num_n_pairs_per_sample=num_n_pairs_per_sample,
            is_image_dataset=False,
        )

        contrastive_val_dataset = ContrastiveDataset(
            supervised_val_dataset,
            rnd=rnd,
            num_n_pairs_per_sample=num_n_pairs_per_sample,
            is_image_dataset=False,
        )

        train_data_loader = DataLoader(
            contrastive_dataset, batch_size=batch_size, drop_last=True, shuffle=True
        )
        validation_data_loader = DataLoader(
            contrastive_val_dataset, batch_size=batch_size, drop_last=True, shuffle=True
        )
        supervised_train_data_loader = DataLoader(
            supervised_train_dataset, batch_size=batch_size, shuffle=False
        )
        supervised_validation_data_loader = DataLoader(
            supervised_val_dataset, batch_size=batch_size, shuffle=False
        )

        model = ContrastiveModel(
            num_hidden=cfg["architecture"]["num_hidden"],
            num_layers=cfg["architecture"]["num_layers"],
            num_last_units=cfg["architecture"]["num_last_units"],
        )
        model = model.to(device)
        contrastive_loss = ContrastiveLoss(
            device=device,
            num_negatives=neg_size,
            normalize=cfg["loss"]["normalize"],
            replace=cfg["loss"]["replace"],
        )

        if cfg["optimizer"]["name"] == "sgd":
            optimizer = torch.optim.SGD(
                params=model.parameters(),
                lr=cfg["optimizer"]["lr"],
                momentum=cfg["optimizer"]["momentum"],
                nesterov=cfg["optimizer"]["nesterov"],
                weight_decay=cfg["optimizer"]["weight_decay"],
            )
        elif cfg["optimizer"]["name"] == "adam":
            optimizer = torch.optim.Adam(
                params=model.parameters(),
                lr=cfg["optimizer"]["lr"],
                betas=(cfg["optimizer"]["beta_0"], cfg["optimizer"]["beta_1"]),
                eps=cfg["optimizer"]["eps"],
                weight_decay=cfg["optimizer"]["weight_decay"],
            )
        elif cfg["optimizer"]["name"] == "adamw":
            optimizer = torch.optim.AdamW(
                params=model.parameters(), lr=cfg["optimizer"]["lr"]
            )
        else:
            raise ValueError("Unsupported optimizer")

        if cfg["lr_scheduler"]["name"] == "plateau":
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, patience=cfg["lr_scheduler"]["patience"]
            )
        else:
            raise ValueError("Unsupported learning scheduler")

        for epoch in range(epochs):
            model.train()
            sum_loss = 0.0

            for anchors, positives in train_data_loader:
                # instead of optimizer.zero_grad()
                # ref: https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#use-parameter-grad-none-instead-of-model-zero-grad-or-optimizer-zero-grad
                for param in model.parameters():
                    param.grad = None

                anchors_features = model(anchors.to(device))  # (B, num_features)
                positives_features = model(positives.to(device))  # (B, num_features)

                loss = contrastive_loss.forward(
                    anchors_features, positives_features
                )  # (,)
                loss.backward()
                optimizer.step()

                sum_loss += loss.item() * len(anchors)

            train_loss_epoch = sum_loss / (
                len(train_data_loader) * train_data_loader.batch_size
            )
            lr_scheduler.step(train_loss_epoch)

            wandb.log(data={"contrastive_train_loss": train_loss_epoch}, step=epoch)

            with torch.no_grad():
                (
                    train_acc,
                    val_acc,
                    train_sup_loss,
                    val_sup_loss,
                    val_cont_loss,
                ) = validation(
                    model,
                    supervised_train_data_loader,
                    supervised_validation_data_loader,
                    validation_data_loader,
                    neg_size,
                    cfg["normalize_eval"],
                    device,
                    cfg["loss"]["replace"],
                )
                logger.info(
                    "{} {:.7f} {:.1f} {:.1f} {:.4f} {:.4f}".format(
                        epoch,
                        train_loss_epoch,
                        train_acc,
                        val_acc,
                        train_sup_loss,
                        val_sup_loss,
                    )
                )

                results[num_latent_classes][neg_size]["contrastive_loss"].append(
                    train_loss_epoch
                )
                results[num_latent_classes][neg_size]["train_acc_history"].append(
                    train_acc
                )
                results[num_latent_classes][neg_size]["val_acc_history"].append(val_acc)
                results[num_latent_classes][neg_size]["train_sup_loss"].append(
                    train_sup_loss
                )
                results[num_latent_classes][neg_size]["val_sup_loss"].append(
                    val_sup_loss
                )
                results[num_latent_classes][neg_size]["val_cont_loss"].append(
                    val_cont_loss
                )

                wandb.log(data={"contrastive_val_loss": val_cont_loss}, step=epoch)
                wandb.log(data={"supervised_val_loss": val_sup_loss}, step=epoch)
                wandb.log(data={"supervised_val_acc": val_acc}, step=epoch)

                if cfg["save_representation_plots"]:
                    plot_representation(
                        supervised_validation_data_loader, model, device, epoch
                    )

    save_results(cfg, results)


if __name__ == "__main__":
    main()
