from abc import ABC
import dataclasses

from typing import Any, Optional
import torch


@dataclasses.dataclass
class ForwardArgs(ABC):
    hidden_states: torch.Tensor

    @classmethod
    def fields(cls) -> list[str]:
        return [f.name for f in dataclasses.fields(cls)]

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.fields()}

    def print_shapes(self) -> None:
        shape_strs = []
        for name, field in self.to_dict().items():
            if field is not None:
                if hasattr(field, "shape"):
                    shape_strs.append(f"{name}.shape: {field.shape}")
                else:
                    shape_strs.append(f"{name}: {field}")
            else:
                shape_strs.append(f"{name}.shape: None")
        print("\n".join(shape_strs))


@dataclasses.dataclass
class PixArtForwardArgs(ForwardArgs):
    attention_mask: Optional[torch.Tensor]
    encoder_hidden_states: Optional[torch.Tensor]
    encoder_attention_mask: Optional[torch.Tensor]
    timestep: Optional[torch.LongTensor]
    cross_attention_kwargs: dict[str, Any]


@dataclasses.dataclass
class FluxForwardArgs(ForwardArgs):
    # encoder_hidden_states used for FluxTransformerBlock, but not FluxSingleTransformerBlock
    encoder_hidden_states: Optional[torch.Tensor]
    temb: torch.FloatTensor
    image_rotary_emb: torch.Tensor
