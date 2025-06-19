import json
from pathlib import Path
from typing import Any

import torch
from ecad.graph.flux_builder import FluxTransformerGraphBuilder
from ecad.schedulers.dit_scheduler.dit_schedule import DiTSchedule

from torch import nn

from ecad.utils import FluxForwardArgs


class FluxDiTSchedule(
    DiTSchedule[FluxForwardArgs, FluxTransformerGraphBuilder]
):
    def __init__(
        self,
        num_blocks: int,
        num_inference_steps: int,
        name: str,
        schedule: dict[int, FluxTransformerGraphBuilder],
        top_level_config: dict[str, Any] | None = None,
        attributes: dict[str, Any] | None = None,
        metrics: dict[str, Any] | None = None,
        transformer_blocks: nn.ModuleList | None = None,
        num_single_blocks: int = None,
        single_transformer_blocks: nn.ModuleList | None = None,
    ) -> None:

        super().__init__(
            num_blocks,
            num_inference_steps,
            name,
            schedule,
            top_level_config=top_level_config,
            attributes=attributes,
            metrics=metrics,
            transformer_blocks=transformer_blocks,
        )

        self.single_transformer_blocks = single_transformer_blocks
        self.num_single_blocks = num_single_blocks

    def update_transformer_blocks(
        self,
        transformer_blocks: nn.ModuleList,
        single_transformer_blocks: nn.ModuleList = None,  # type: ignore
        **kwargs: Any,
    ) -> None:
        if single_transformer_blocks is None:
            raise ValueError("must pass valid single_transformer_blocks")

        self.transformer_blocks = transformer_blocks
        self.single_transformer_blocks = single_transformer_blocks
        for builder in self.schedule.values():
            builder.update_transformer_blocks(
                transformer_blocks=self.transformer_blocks,
                single_transformer_blocks=self.single_transformer_blocks,
            )

    def forward(
        self,
        forward_args: FluxForwardArgs,
        inference_step: int,
        transformer_blocks: nn.ModuleList | None = None,
        single_transformer_blocks: nn.ModuleList = None,  # type: ignore
        **kwargs: Any,
    ) -> torch.Tensor:
        if (
            transformer_blocks is not None
            and single_transformer_blocks is not None
        ):
            self.update_transformer_blocks(
                transformer_blocks=transformer_blocks,
                single_transformer_blocks=single_transformer_blocks,
            )

        if (
            self.transformer_blocks is None
            or self.single_transformer_blocks is None
        ):
            raise ValueError(
                "transformer_blocks and single_transformer_blocks must be set before calling forward."
            )

        # TODO: for now, try adding a short circuit here to do a normal forward pass
        # to speed up generation
        builder = self.schedule[inference_step]
        return builder.forward(forward_args)

    def to_dict(self) -> dict[str, Any]:
        data = super().to_dict()
        data["num_single_blocks"] = self.num_single_blocks
        return data

    @classmethod
    def from_json(cls, file_path: Path) -> "FluxDiTSchedule":
        """Load the schedule from a JSON file at the specified file path."""
        with file_path.open("r") as f:
            data = json.load(f)

        top_level_config = data.get("config", None)
        metrics = data.get("metrics", None)
        data = data["dit_schedule"]

        num_blocks = data["num_blocks"]
        num_single_blocks = data["num_single_blocks"]
        num_inference_steps = data["num_inference_steps"]
        name = data["name"]
        attributes = data.get("attributes", None)

        placeholder_blocks = nn.ModuleList(
            (nn.Identity() for _ in range(num_blocks))
        )

        placeholder_single_blocks = nn.ModuleList(
            (nn.Identity() for _ in range(num_single_blocks))
        )

        schedule = {
            int(step_num): FluxTransformerGraphBuilder(
                json_config=inner,
                forward_args_type=FluxForwardArgs,
                num_blocks=num_blocks,
                num_single_blocks=num_single_blocks,
                transformer_blocks=placeholder_blocks,
                single_transformer_blocks=placeholder_single_blocks,
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
            num_single_blocks=num_single_blocks,
        )

    def visualize_schedule(self, output_dir: Path) -> None:
        """Visualize the schedule's graph as a png for each timestep."""

        for step, builder in self.schedule.items():
            fname = output_dir / f"{self.name}_{step:03}.png"
            builder.visualize_fx_graph(fname, True)
