from functools import reduce
from operator import and_

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import style
import wandb
from plot_k_msuploss import lb_intercept, nceloss_min, suploss_min, ub_intercept


def make_segments(x, y):
    """
    Create list of line segments from x and y coordinates, in the correct format
    for LineCollection: an array of the form numlines x (points per line) x 2 (x
    and y) array
    """

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments


def colorline(
    x,
    y,
    z=None,
    cmap=plt.get_cmap("copper"),
    norm=plt.Normalize(0.0, 1.0),
    linewidth=3,
    alpha=1.0,
):
    """
    http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
    http://matplotlib.org/examples/pylab_examples/multicolored_line.html
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    """

    # Default colors equally spaced on [0,1]:
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))

    # Special case if a single number:
    if not hasattr(z, "__iter__"):  # to check for numerical input -- this is a hack
        z = np.array([z])

    z = np.asarray(z)

    segments = make_segments(x, y)
    lc = matplotlib.collections.LineCollection(
        segments, array=z, cmap=cmap, norm=norm, linewidth=linewidth, alpha=alpha
    )

    ax = plt.gca()
    ax.add_collection(lc)

    return lc


def filter_runs(runs, filters):
    ret = []
    _test = lambda it: it[0] in run.config and run.config[it[0]] == it[1]
    for run in runs:
        if reduce(and_, map(_test, filters.items()), True):
            ret.append(run)
    return ret


def sort_runs(runs, key):
    return sorted(runs, key=lambda it: it.config[key])


def plot_trajectory(runs, filename, seed=7):
    style.sns.set(font_scale=1.5)

    plt.figure(figsize=(18, 2.5))
    c = 100

    target_runs = sort_runs(filter_runs(runs, {"seed": seed}), "loss.neg_size")

    base_width = 10
    grid = plt.GridSpec(1, base_width * len(target_runs) + 1, wspace=1.0)

    for i, run in enumerate(target_runs):
        k = run.config["loss.neg_size"]
        history = run.history()
        trj_x = history["contrastive_val_loss"]
        trj_y = history["supervised_val_loss"]
        x = np.linspace(0, 12, 1000)
        x0 = nceloss_min([k], c)
        y0 = suploss_min(c)

        plt.subplot(1, 5, i + 1)
        ax = plt.subplot(grid[0, (base_width * i) : (base_width * (i + 1))])
        lb = x + lb_intercept(k, c)
        ub = x + ub_intercept(k, c)
        plt.plot(x, ub, "b-", label=r"$R_\mathrm{cont} + \Delta_\mathrm{U}$")
        plt.plot(x, lb, "b-.", label=r"$R_\mathrm{cont} + \Delta_\mathrm{L}$")
        plt.plot([x0 - 1e-3, x0 + 1e-3], [0, 10], "k:")
        plt.plot(x, y0 * np.ones_like(x), "k:")
        colorline(
            trj_x,
            trj_y,
            z=np.linspace(0, 1, len(trj_x)) ** 0.4,
            linewidth=3,
            alpha=0.8,
            cmap=plt.get_cmap("spring_r"),
        )

        plt.fill_between(
            x, np.maximum(lb, y0), ub, where=(x >= x0), facecolor="#aaaaff", alpha=0.8
        )

        plt.title(fr"$K = {k}$")
        plt.xticks([0, 2, 4, 6])
        plt.xlabel(r"contrastive loss")
        plt.xlim(0, 8)
        plt.ylim(2, 6)
        if i == 0:
            max_epoch = len(trj_x)
            plt.ylabel(r"mean supervised loss")
            plt.legend(loc="lower right")
        else:
            ax.set_yticklabels([])

    # color bar
    ax = plt.subplot(grid[0, -1:])
    norm = matplotlib.colors.Normalize(vmin=0, vmax=max_epoch)
    bounds = max_epoch * (np.linspace(0, 1, 100) ** (1 / 0.4))
    colorbar = matplotlib.colorbar.ColorbarBase(
        ax,
        cmap=plt.get_cmap("spring_r"),
        norm=norm,
        boundaries=bounds,
        ticks=[0, 20, 100, 300],
    )
    colorbar.set_label("Trajectory (epoch)")

    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight", pad_inches=0.1)
    style.sns.set(font_scale=1.0)


def plot_learning_curve(runs, filename):
    style.sns.set(font_scale=1.2)

    plt.figure(figsize=(15, 3))
    epochs = 300
    ks = np.array([1, 4, 16, 64, 256])

    # (val) supervised loss plot
    plt.subplot(1, 3, 1)
    x = np.arange(epochs)
    for i, k in enumerate(ks):
        _runs = filter_runs(runs, {"loss.neg_size": k})
        ys = np.array([_run.history()["supervised_val_loss"] for _run in _runs])
        mean = ys.mean(axis=0)
        std = ys.std(axis=0)
        plt.plot(x, mean, label=fr"$K = {k}$")
        plt.fill_between(x, mean - std, mean + std, alpha=0.3)
    plt.xlabel("epoch")
    plt.ylabel("mean supervised loss")
    plt.ylim(3.6, 4.5)
    plt.legend(loc="upper right")

    # (val) supervised accuracy plot
    plt.subplot(1, 3, 2)
    x = np.arange(epochs)
    for k in ks:
        _runs = filter_runs(runs, {"loss.neg_size": k})
        ys = np.array([_run.history()["supervised_val_acc"] for _run in _runs])
        mean = ys.mean(axis=0)
        std = ys.std(axis=0)
        plt.plot(x, mean, label=f"K = {k}")
        plt.fill_between(x, mean - std, mean + std, alpha=0.3)
    plt.xlabel("epoch")
    plt.ylabel("mean supervised accuracy")

    # best val loss comparison
    plt.subplot(1, 3, 3)
    loss = []
    for k in ks:
        _runs = filter_runs(runs, {"loss.neg_size": k})
        ys = np.array([_run.history()["supervised_val_loss"] for _run in _runs])
        loss.append(ys.min(axis=1))
    mean = np.mean(loss, axis=1)
    std = np.std(loss, axis=1)
    plt.errorbar(ks, mean, yerr=std, fmt="o", markersize=5, capsize=5)
    plt.xscale("log")
    ax = plt.gca()
    plt.tick_params(axis="x", which="minor")
    ax.xaxis.set_minor_formatter(matplotlib.ticker.FormatStrFormatter("%d"))
    ax.xaxis.set_major_formatter(matplotlib.ticker.NullFormatter())
    ax.xaxis.set_minor_locator(matplotlib.ticker.LogLocator(base=4))
    ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
    plt.xlabel(r"$K$")
    plt.ylabel("best mean supervised loss")

    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight", pad_inches=0.1)

    style.sns.set(font_scale=1.0)


if __name__ == "__main__":
    # get runs
    api = wandb.Api()
    runs = api.runs("INPUT_YOUR_ENTITY/CURL", filters={"tags": "circle"})
    filters = {
        "dataset.name": "circle",
        "dataset.num_latent_classes": 100,
        "optimizer.lr": 0.01,
        "optimizer.mini_batch_size": 1024,
    }
    runs = filter_runs(runs, filters)

    plot_trajectory(runs, "toy_trajectory.pdf")

    plot_learning_curve(runs, "loss_and_curve.pdf")
