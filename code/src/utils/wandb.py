import os

import pandas
from omegaconf import OmegaConf


def flatten_omegaconf(cfg: OmegaConf) -> dict:
    """
    :param cfg: Hydra's config instance.

    flatten nested Hydra's OmegaConf instance to convert OmegaConf to primitive dict for W&B.
    https://stackoverflow.com/a/41801708/2406562
    """

    _config = {
        k.replace("_content", ""): v for k, v in OmegaConf.to_container(cfg).items()
    }
    _config = pandas.json_normalize(_config, sep=".")
    _config = _config.to_dict(orient="records")[0]
    _config["hydra_path"] = os.getcwd()

    return _config
