#!/usr/bin/env python
# coding: utf-8

import matplotlib as mpl

pgf_with_rc_fonts = {
    "font.serif": [],  # use latex default serif font
    "font.sans-serif": ["DejaVu Sans"],  # use a specific sans-serif font
    "font.size": 12,
    "ps.useafm": True,
    "pdf.use14corefonts": True,
    "text.usetex": True,
}
mpl.rcParams.update(pgf_with_rc_fonts)

import seaborn as sns

sns.set()
