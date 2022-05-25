import os

import hydra
import omegaconf
from torch import distributed
from torch.distributed.elastic.multiprocessing.errors import record


@record
@hydra.main(config_path=None, config_name="main")
def main(cfg: omegaconf.DictConfig):
    local_rank = int(os.environ["LOCAL_RANK"])
    distributed.init_process_group(backend="nccl")


if __name__ == "__main__":
    main()
