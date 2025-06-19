from abc import ABC, abstractmethod
import json
from pathlib import Path
from typing import Any, Dict, Generic, TypeVar

from torch import nn
import torch

from ecad.graph.pixart_builder import TransformerGraphBuilder
from ecad.utils import ForwardArgs


ForwardArgsT = TypeVar("ForwardArgsT", bound=ForwardArgs)
TransformerGraphBuilderT = TypeVar(
    "TransformerGraphBuilderT", bound=TransformerGraphBuilder
)


class DiTSchedule(ABC, Generic[ForwardArgsT, TransformerGraphBuilderT]):
    def __init__(
        self,
        num_blocks: int,
        num_inference_steps: int,
        name: str,
        schedule: dict[int, TransformerGraphBuilderT],
        top_level_config: dict[str, Any] | None = None,
        attributes: dict[str, Any] | None = None,
        metrics: dict[str, Any] | None = None,
        transformer_blocks: nn.ModuleList | None = None,
        **kwargs: Any,
    ) -> None:
        self.num_blocks: int = num_blocks
        self.num_inference_steps: int = num_inference_steps
        self.name: str = name
        self.schedule: dict[int, TransformerGraphBuilderT] = schedule
        self.transformer_blocks: nn.ModuleList | None = None
        self.top_level_config = (
            top_level_config if top_level_config is not None else {}
        )
        self.attributes = attributes if attributes is not None else {}
        self.metrics = metrics if metrics is not None else {}

    @abstractmethod
    def update_transformer_blocks(
        self, transformer_blocks: nn.ModuleList, **kwargs: Any
    ) -> None:
        pass

    def compile(self, mode: str = "max-autotune") -> None:
        print("WARNING: compilation is extremely slow and likely worth it.")

        # extremely slow, for negligible, if any, speedup
        for step, builder in self.schedule.items():
            self.schedule[step] = torch.compile(
                builder, mode=mode, fullgraph=True
            )

    @abstractmethod
    def forward(
        self,
        forward_args: ForwardArgsT,
        inference_step: int,
        transformer_blocks: nn.ModuleList | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        pass

    def to_dict(self) -> dict[str, Any]:
        data = {
            "dit_schedule": {
                "num_blocks": self.num_blocks,
                "num_inference_steps": self.num_inference_steps,
                "name": self.name,
                "attributes": self.attributes,
                "schedule": {
                    f"{step:03}": builder.to_json()
                    for step, builder in self.schedule.items()
                },
            },
            "config": self.top_level_config,
            "metrics": self.metrics,
        }
        return data

    def to_json(self, file_path: Path) -> None:
        """Save the schedule as a JSON file at the specified file path."""
        data = self.to_dict()
        with file_path.open("w") as f:
            # don't sort since they are created in sorted order
            # and a lexographic sort here breaks ordering
            json.dump(data, f, indent=4, sort_keys=False)

    @classmethod
    @abstractmethod
    def from_json(cls, file_path: Path) -> "DiTSchedule":
        """Load the schedule from a JSON file at the specified file path."""
        pass

    def visualize_schedule(self, output_dir: Path) -> None:
        """Visualize the schedule's graph as a png for each timestep."""
        print("WARNING: no visualization implemented.")
