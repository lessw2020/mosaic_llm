import argparse
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from llm.src.mosaic_gpt import GPTBlock, TorchCausalAttention
from omegaconf import DictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.tensor.parallel.fsdp import is_available


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


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    print(f" rank {rank} has joined")


def cleanup():
    dist.destroy_process_group()


def demo_2d(rank, args):
    """
    Main body of the demo of a basic version of 2d parallel by using
    PyTorch native APIs.
    """
    setup(rank, args.world_size)
    # 2-D mesh is [dp, tp]
    twod_mesh = DeviceMesh(
        device_type="cuda",
        mesh=torch.arange(0, args.world_size).view(args.model_parallel_size, -1),
    )
    gpt_config = DictConfig(
        {
            "alibi": False,
            "attn_impl": "torch",  # Only support torch now
            "d_model": 1024,
            "n_heads": 8,
            "attn_pdrop": 0.01,
            "mlp_ratio": 4,
            "resid_pdrop": 0.01,
        }
    )
    model = GPTBlock(gpt_config, causal_attn_cls=TorchCausalAttention).cuda(rank)
    model = parallelize_module(
        model,
        twod_mesh,
        {"causal_attn": PairwiseParallel(), "mlp": PairwiseParallel()},
        tp_mesh_dim=1,
    )
    fsdp_pg = twod_mesh.get_dim_groups()[0]
    # verify fsdp
    fsdp_is_available = is_available()
    assert fsdp_is_available, "FSDP is not available but required!"

    model = FSDP(model, process_group=fsdp_pg)
    # Create a optimizer for the parallelized module.
    LR = 0.25
    _optimizer = torch.optim.SGD(model.parameters(), lr=LR)
    # Perform a num of iterations of forward/backward
    # and optimizations for the sharded module.
    cleanup()


def run_demo(demo_fn, args):
    mp.spawn(demo_fn, args=(args,), nprocs=args.world_size, join=True)


if __name__ == "__main__":
    n_gpus = torch.cuda.device_count()
    parser = argparse.ArgumentParser()
    # This is passed in via cmd
    parser.add_argument("--world_size", type=int, default=n_gpus)
    parser.add_argument("--iter_nums", type=int, default=10)
    parser.add_argument("--model_parallel_size", type=int, default=2)
    args = parser.parse_args()
    # The main entry point is called directly without using subprocess
    if n_gpus < 2:
        print("Requires at least 2 GPUs to run.")
    elif not TP_AVAILABLE:
        print(
            "PyTorch doesn't have Tensor Parallelism available," " need nightly build."
        )
    else:
        run_demo(demo_2d, args)
