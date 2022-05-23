import torch
import torch.distributed as dist
from omegaconf import OmegaConf


def init_ddp(cfg: OmegaConf, local_rank: int, world_size: int = 4) -> None:
    """
    Initialize the pytorch's DDP config.

    :param cfg: hydra's config.
    :param local_rank: local rank, should be [0, world_size-1].
    :param world_size: the number of GPUs
    """
    dist_url = cfg["distributed"]["dist_url"]

    torch.cuda.set_device(local_rank)

    # prepare distributed
    dist.init_process_group(
        backend="nccl", init_method=dist_url, world_size=world_size, rank=local_rank,
    )

    # https://github.com/facebookresearch/maskrcnn-benchmark/issues/172
    dist.barrier()


def cleanup() -> None:
    dist.destroy_process_group()
