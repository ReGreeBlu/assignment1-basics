import torch
import os
import typing

def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                    iteration: int, out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]):
    model_state = model.state_dict()
    optimizer_state = optimizer.state_dict()
    obj = {"model_state": model_state, "optimizer_state": optimizer_state, "iteration": iteration}
    torch.save(obj, out)

def load_checkpoint(src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
                    model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer):
    obj = torch.load(src)
    model.load_state_dict(obj["model_state"])
    optimizer.load_state_dict(obj["optimizer_state"])
    iteration = obj["iteration"]
    return iteration

