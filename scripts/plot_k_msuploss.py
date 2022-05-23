#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import numpy as np
import style
from scipy.special import comb
from scipy.stats import entropy


def lb_intercept(k, c=100, b=1):
    pi = np.ones(c) / c
    gamma = entropy(pi) + np.log(k) - 2 * np.log(1 + k) - 2 * np.log(np.cosh(b ** 2))
    return gamma


def ub_intercept(k, c=100, b=1):
    pi = np.ones(c) / c
    delta = np.log(pi.max() * c ** 2 * np.cosh(b ** 2) ** 2 / k)
    return delta


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


def suploss_min(c, b=1):
    suploss_min = np.log(1 + (c - 1) * np.exp(-2 * b ** 2))
    return suploss_min


def plot_curve(k, c, color):
    nceloss = nceloss_min(k, c)
    suploss = suploss_min(c) * np.ones_like(k)
    b1 = ub_intercept(k, c)
    plt.plot(
        k,
        b1 + nceloss,
        "-",
        lw=3,
        color=color,
        label=r"$R_\mathrm{cont}^* + \Delta_\mathrm{U}$ ($C=" + str(c) + r"$)",
    )
    plt.plot(
        k,
        suploss,
        ":",
        lw=3,
        color=color,
        label=r"$R_{\mu\mathchar`-\mathrm{supv}}^*$ ($C=" + str(c) + r"$)",
    )


if __name__ == "__main__":
    k = np.arange(1, 100, 1)

    plt.figure(figsize=(5, 3))
    plot_curve(k, c=50, color="C1")
    plot_curve(k, c=100, color="C2")

    plt.xlabel(r"$K$")
    plt.legend()

    plt.tight_layout()
    plt.savefig("k_msuploss.pdf", bbox_inches="tight", pad_inches=0.1)
