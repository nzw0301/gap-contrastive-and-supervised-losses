#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import numpy as np
import style
from plot_k_msuploss import lb_intercept, nceloss_min, suploss_min


def plot_curve(k, c, color):
    nceloss = nceloss_min(k, c)
    suploss = suploss_min(c) * np.ones_like(k)
    b0 = lb_intercept(k, c)
    plt.plot(
        k,
        suploss - b0,
        "-",
        lw=3,
        color=color,
        label=r"$R_{\mu\mathchar`-\mathrm{supv}}^* - \Delta_\mathrm{L}$ ($C="
        + str(c)
        + r"$)",
    )
    plt.plot(
        k,
        nceloss,
        ":",
        lw=3,
        color=color,
        label=r"$R_\mathrm{cont}^*$ ($C=" + str(c) + r"$)",
    )


if __name__ == "__main__":
    k = np.arange(1, 100, 1)

    plt.figure(figsize=(5, 3))
    plot_curve(k, c=50, color="C1")
    plot_curve(k, c=100, color="C2")

    plt.xlabel(r"$K$")
    plt.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig("k_nceloss.pdf", bbox_inches="tight", pad_inches=0.1)
