#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import numpy as np
import style
from plot_k_msuploss import nceloss_min, ub_intercept
from scipy.special import comb


def coupon_collector_probability(k, c):
    # func2 on http://aquarius10.cse.kyutech.ac.jp/~otabe/shokugan/sg2.html
    ret = []
    for _k in k:
        p = np.zeros(c + 1)
        p[0] = 1
        for j in range(_k):
            for i in range(c, -1, -1):
                p[i] = p[i] * i / c + p[i - 1] * (c - i + 1) / c
                if i == 0:
                    p[0] = 0
        ret.append(p[c])

    return np.array(ret)


def collision_term(k, c):
    ret = []
    for _k in k:
        r = np.sum(
            [
                (1 / c) ** m * (1 - 1 / c) ** (_k - m) * comb(_k, m) * np.log(1 + m)
                for m in range(_k + 1)
            ]
        )
        ret.append(r)
    return np.array(ret)


if __name__ == "__main__":
    c = 10
    k = np.arange(1, 100, 1)
    plt.figure(figsize=(4, 2))

    # ====================================
    # upper bound coefficient
    # ====================================
    tau = 1 - (1 - 1 / c) ** k
    v = coupon_collector_probability(k + 1, c)
    harmonic = lambda c: np.log(c) + 0.577
    coef = np.ones_like(k)
    coef_arora = 1 / ((1 - tau) * v)
    coef_nozawa = 2 / v
    coef_ash = 2 * np.maximum(1, 2 * (c - 1) * harmonic(c - 1) / k) / (1 - tau)

    # plt.title(f"C = {c}")
    plt.xlabel(r"$K$")
    # plt.ylabel(r"coefficient of $R_\mathrm{cont}$")
    plt.plot([c - 0.1, c + 0.1], [-1e10, 1e10], "k:")
    plt.plot(k, coef, label=r"Ours")
    plt.plot(k[c:], coef_arora[c:], label=r"Arora et al.")
    plt.plot(k[c:], coef_nozawa[c:], label=r"Nozawa \& Sato")
    plt.plot(k, coef_ash, label=r"Ash et al.")
    plt.yscale("log")
    plt.ylim((0.5, 500))
    plt.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig("upper_bound_coef.pdf", bbox_inches="tight", pad_inches=0.1)

    plt.clf()

    # ====================================
    # upper bound comparison
    # ====================================
    nceloss = nceloss_min(k, c)
    collision = collision_term(k, c)
    ub = nceloss + ub_intercept(k, c)
    ub_arora = coef_arora * (nceloss - collision)
    ub_nozawa = (2 * nceloss - collision) / v
    ub_ash = coef_ash * (nceloss - collision)

    plt.xlabel(r"$K$")
    plt.plot([c - 0.1, c + 0.1], [-1e10, 1e10], "k:")
    plt.plot(k, ub, label=r"Ours")
    plt.plot(k[c:], ub_arora[c:], label=r"Arora et al.")
    plt.plot(k[c:], ub_nozawa[c:], label=r"Nozawa \& Sato")
    plt.plot(k, ub_ash, label=r"Ash et al.")
    plt.yscale("log")
    plt.ylim((0.5, 500))

    plt.tight_layout()
    plt.savefig("upper_bound.pdf", bbox_inches="tight", pad_inches=0.1)
