from typing import Callable

import torch


class _FunctionDescWrapper:
    def __init__(self, func, desc):
        self.func = func
        self.desc = desc
        self.__name__ = desc

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    def __repr__(self):
        return self.desc


def identity(x):
    return x


def add(*tensors):
    return torch.sum(torch.stack(tensors), dim=0)


def avg(*tensors):
    return torch.mean(torch.stack(tensors, dim=0), dim=0)


FUNC_REGISTRY: dict[str, Callable] = {
    "identity": identity,
    "add": add,
    "avg": avg,
}
DEFAULT_FUNC_NAME = "identity"

for func_name, func in FUNC_REGISTRY.items():
    FUNC_REGISTRY[func_name] = _FunctionDescWrapper(func, func_name)
