import json
from pathlib import Path
from typing import Any

import torch
from torch import nn

from ecad.graph.pixart_builder import PixArtTransformerGraphBuilder
from ecad.schedulers.dit_scheduler.dit_schedule import DiTSchedule
from ecad.utils import PixArtForwardArgs


class PixArtDiTSchedule(
    DiTSchedule[PixArtForwardArgs, PixArtTransformerGraphBuilder]
):

    def update_transformer_blocks(
        self, transformer_blocks: nn.ModuleList, **kwargs: Any
    ) -> None:
        self.transformer_blocks = transformer_blocks
        for builder in self.schedule.values():
            builder.update_transformer_blocks(self.transformer_blocks)

    def forward(
        self,
        forward_args: PixArtForwardArgs,
        inference_step: int,
        transformer_blocks: nn.ModuleList | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        if transformer_blocks is not None:
            self.update_transformer_blocks(transformer_blocks)

        if self.transformer_blocks is None:
            raise ValueError(
                "transformer_blocks must be set before calling forward."
            )

        builder = self.schedule[inference_step]
        return builder.forward(forward_args)

    @classmethod
    def from_json(cls, file_path: Path) -> "PixArtDiTSchedule":
        """Load the schedule from a JSON file at the specified file path."""
        with file_path.open("r") as f:
            data = json.load(f)

        top_level_config = data.get("config", None)
        metrics = data.get("metrics", None)
        data = data["dit_schedule"]

        num_blocks = data["num_blocks"]
        num_inference_steps = data["num_inference_steps"]
        name = data["name"]
        attributes = data.get("attributes", None)

        placeholder_blocks = nn.ModuleList(
            (nn.Identity() for _ in range(num_blocks))
        )

        schedule = {
            int(step_num): PixArtTransformerGraphBuilder(
                inner, PixArtForwardArgs, placeholder_blocks
            )
            for step_num, inner in data["schedule"].items()
        }

        return cls(
            num_blocks=num_blocks,
            num_inference_steps=num_inference_steps,
            name=name,
            schedule=schedule,
            top_level_config=top_level_config,
            attributes=attributes,
            metrics=metrics,
        )

    def visualize_schedule(self, output_dir: Path) -> None:
        """Visualize the schedule's graph as a png for each timestep."""

        for step, builder in self.schedule.items():
            fname = output_dir / f"{self.name}_{step:03}.png"
            builder.visualize_fx_graph(fname, False)
