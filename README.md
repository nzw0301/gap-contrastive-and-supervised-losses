# Experimental code of [`On the Surrogate Gap between Contrastive and Supervised Losses`](https://proceedings.mlr.press/v162/bao22e.html)

## System-related versions

- python: 3.6.8
- CUDA: 11.2
- cudnn: 8005

## Create an experimental environment

```bash
pip install -r requirements.txt

git clone git@github.com:NVIDIA/apex.git
cd apex
git checkout 54b93919aadc117cbab1fe5a2af4664bb9842928
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

Install an optional library for efficient experiments.

```bash
# Install gnu-parallel from source.
wget https://ftp.gnu.org/gnu/parallel/parallel-20210622.tar.bz2
tar -jxvf parallel-20210622.tar.bz2
cd parallel-20210622
./configure --prefix=~/bin
make
make install
```

## Execute training code

- Toy: Please run [`code/jobs/toy/circle.sh`](code/jobs/toy/circle.sh).
- Vision: Please run `{cifar10,cifar100}/contrastive/seed_*.sh` under [`code/jobs/vision`](./code/jobs/vision) directory.
- Language: Please run [`code/jobs/language/wiki3029/contrastive.sh`](code/jobs/language/wiki3029/contrastive.sh).

Note that this repo manages the experimental results by using [`wandb`](https://wandb.ai/). Please replace `INPUT_YOUR_ENTITY` with your wandb username in codes that you would like to run.

## Evaluation

### Preparation

Please run [`./code/notebooks/extract_eval_weights_from_wandb.ipynb`](code/notebooks/extract_eval_weights_from_wandb.ipynb).

### Perform evaluation to draw plots


- Vision: Please run the `{cifar10,cifar100}/mean_eval.sh`, `{cifar10,cifar100}/linear_eval.sh` and `{cifar10,cifar100}/contrastive_eval.sh` scripts under [`code/jobs/vision`](code/jobs/vision) directory.
- Language: Please run the `mean_eval.sh`, `linear_eval.sh` and `contrastive_eval.sh` scripts under [`code/jobs/language/wiki3029`](code/jobs/language/wiki3029) directory.

### Create plots

Please run the following notebooks to generate plots

#### Synthetic experiments

- [`scripts/plot_k_msuploss.py`](scripts/plot_k_msuploss.py): Figure 3 (a)
- [`scripts/plot_k_nceloss.py`](scripts/plot_k_nceloss.py): Figure 3 (b)
- [`scripts/compare_upper_bound.py`](scripts/compare_upper_bound.py): Figure 4
- [`scripts/plot_toy_trajectory.py`](scripts/plot_toy_trajectory.py): Figure 5
- [`scripts/make_toy_figure.py`](scripts/make_toy_figure.py): Figure 6

Note that the scripts for Figures 5 and 6 require the results generated by [`code/jobs/toy/circle.sh`](code/jobs/toy/circle.sh).


#### Real benchmark datasets

- [`code/notebooks/bound.ipynb`](code/notebooks/bound.ipynb): Figures 1 and 9
- [`code/notebooks/ck_heatmap.ipynb`](code/notebooks/ck_heatmap.ipynb) Figures 7 and 8

---


**Note:**: this codebase tracks experiments using Weights & Biases, but the default wandb might cause [hanging at the beginning of training]((https://docs.wandb.ai/guides/track/advanced/distributed-training#hanging-at-the-beginning-of-training)) of distributed training code. To avoid this, please set an environment variable as follows:

```bash
WANDB_START_METHOD="thread"
```


---

## References

```
@inproceedings{BNN2022,
    title = {{On the Surrogate Gap between Contrastive and Supervised Losses}},
    author = {Bao, Han and Nagano, Yoshihiro and Nozawa, Kento},
    year = {2022},
    booktitle = {ICML},
    pages = {1585--1606},
}
```

## Related resource

- [PMLR](https://proceedings.mlr.press/v162/bao22e.html)
- [arXiv](https://arxiv.org/abs/2110.02501)
- [Poster](https://hermite.jp/posters/202207_ICML.pdf)
