from abc import ABC
from typing import Callable
import torch


class ComputeRegistry(ABC):
    # Each subclass should override _registry to be its own dictionary.
    _registry: dict[str, Callable[..., torch.Tensor]] = {}

    @classmethod
    def register(
        cls, func: Callable[..., torch.Tensor]
    ) -> Callable[..., torch.Tensor]:
        """Decorator to register a function in the registry.

        The function’s __name__ is used as the key (in lowercase).
        """
        key = func.__name__.lower()
        cls._registry[key] = func
        return func

    @classmethod
    def get(
        cls, key: str | None, none_if_not_found: bool = False
    ) -> Callable[..., torch.Tensor] | None:
        """Returns the function that matches the given key.

        If no match is found, returns None (if none_if_not_found is True)
        or returns the default function (if none_if_not_found is False).
        """
        if key is not None:
            key = key.lower()
            if key in cls._registry:
                return cls._registry[key]
        return None if none_if_not_found else cls.default()

    @classmethod
    def default(cls) -> Callable[..., torch.Tensor]:
        if not hasattr(cls, "DEFAULT"):
            raise NotImplementedError(
                "Subclasses must define a DEFAULT attribute."
            )

        default_func = cls._registry.get(getattr(cls, "DEFAULT"))
        if default_func is None:
            raise ValueError(
                "Default function 'compute_ff_cached' not registered."
            )
        return default_func


class ComputeAttnRegistry(ComputeRegistry):
    _registry: dict[str, Callable[..., torch.Tensor]] = {}
    DEFAULT = "compute_attn_cached"


class ComputeFFRegistry(ComputeRegistry):
    _registry: dict[str, Callable[..., torch.Tensor]] = {}
    DEFAULT = "compute_ff_cached"
