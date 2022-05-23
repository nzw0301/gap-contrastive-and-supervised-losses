#!/usr/bin/env python
# coding: utf-8

import matplotlib as mpl

pgf_with_rc_fonts = {
    "font.size": 10,
}
mpl.rcParams.update(pgf_with_rc_fonts)

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import comb


def nceloss_min(k, c=100, b=1):
    nceloss_min = []
    for _k in k:
        l = np.sum(
            [
                comb(_k, m)
                * (1 / c) ** m
                * (1 - 1 / c) ** (_k - m)
                * np.log(1 + m + (_k - m) * np.exp(-2 * b ** 2))
                for m in range(_k + 1)
            ]
        )
        nceloss_min.append(l)
    nceloss_min = np.array(nceloss_min)

    return nceloss_min


def plot_trajectory(result, output_filename="toy_trajectory.pdf"):
    c = list(result.keys())[0]
    k = list(result[c].keys())[0]
    trj_x = result[c][k]["val_cont_loss"]
    trj_y = result[c][k]["val_sup_loss"]
    c = int(c)
    k = int(k)

    x = np.linspace(0, 12, 1000)
    x0 = nceloss_min([k], c)
    min_trj_y = np.min(trj_y)

    # upper bound
    au = 1
    bu = 2 * np.log(np.cosh(1)) - np.log(k / c)
    ub = au * x + bu

    # lower bound
    al = 1
    bl = np.log(c) + np.log(k / (k + 1) ** 2) - 2 * np.log(np.cosh(1))
    lb = al * x + bl

    min_sup_loss = np.log(1 + (c - 1) * np.exp(-2))
    max_sup_loss = np.log(1 + (c - 1) * np.exp(+2))

    plt.plot(x, lb, "b-", lw=1.5, label=r"Lower bound (Thm 4)")
    plt.plot(x, ub, "b-.", lw=1.5, label=r"Upper bound (Thm 3)")
    plt.plot([x0 - 1e-3, x0 + 1e-3], [0, 10], "b:", lw=1.5)
    plt.plot(
        x, min_sup_loss * np.ones_like(x), "b:", lw=1.5
    )  # NOTE: lower bound of sup loss
    plt.plot(trj_x, trj_y, "k-", lw=3, label=r"trajectory")

    plt.fill_between(
        x,
        np.maximum(lb, min_sup_loss),
        np.minimum(ub, max_sup_loss),
        where=(x >= x0) & (lb <= max_sup_loss),
        facecolor="#aaaaff",
        alpha=0.5,
    )

    plt.title(f"C={c}, K={k} (min sup loss = {min_trj_y:.3f})")
    plt.xlabel(r"contrastive loss")
    plt.ylabel(r"mean supervised loss")
    plt.xlim(0, 8)
    plt.ylim(0, 5)
    plt.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(output_filename, bbox_inches="tight", pad_inches=0.1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "filename",
        help="location of execution log file name",
        default=os.path.dirname(__file__) + "/../code/results.json",
    )
    args = parser.parse_args()

    with open(args.filename, "r") as f:
        result = json.load(f)

    pgf_with_rc_fonts = {
        "font.serif": [],  # use latex default serif font
        "font.sans-serif": ["DejaVu Sans"],  # use a specific sans-serif font
        "font.size": 10,
        "ps.useafm": True,
        "pdf.use14corefonts": True,
        "text.usetex": True,
    }
    mpl.rcParams.update(pgf_with_rc_fonts)

    plot_trajectory(result)
