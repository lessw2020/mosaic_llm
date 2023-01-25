import argparse
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from llm.src.mosaic_gpt import GPTBlock
from omegaconf import DictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

TP_AVAILABLE = False
try:
    from torch.distributed._tensor import (
        DeviceMesh,
    )
    from torch.distributed.tensor.parallel import (
        PairwiseParallel,
        parallelize_module,
    )

    TP_AVAILABLE = True

except BaseException as e:
    print(f"!! Exception during TP init - {e=}\n")
    pass

print(f"TP is available == {TP_AVAILABLE}")
