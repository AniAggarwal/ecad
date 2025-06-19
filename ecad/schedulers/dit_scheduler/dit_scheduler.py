from pathlib import Path
from typing import Any

import torch
from torch import nn

from ecad.schedulers.dit_scheduler.dit_schedule import DiTSchedule
from ecad.utils import ForwardArgs


class DiTScheduler:
    def __init__(self, schedule: DiTSchedule | Path, verbose: bool = False):
        try:
            self.schedule: DiTSchedule = (
                schedule
                if isinstance(schedule, DiTSchedule)
                else DiTSchedule.from_json(schedule)
            )
        except Exception as e:
            raise ValueError(
                f"Failed to load schedule from {schedule}."
            ) from e

        self.reset_step()
        self.verbose: bool = verbose

    @property
    def curr_step(self) -> int:
        return self._last_step + 1

    def reset_step(self) -> None:
        self._last_step = -1

    def update_transformer_blocks(
        self, transformer_blocks: nn.ModuleList, **kwargs: Any
    ) -> None:
        self.schedule.update_transformer_blocks(transformer_blocks, **kwargs)

    def _print(self, *args, **kwargs):
        # TODO: switch to logging at some point
        if self.verbose:
            print(*args, **kwargs)

    def per_step_callback(self, step: int, timestep: int, **kwargs: Any):
        self._print(
            f"Called DiTScheduler callback at step {step}, timestep {timestep}."
        )
        self._last_step = step

    def forward(
        self,
        forward_args: ForwardArgs,
        transformer_blocks: nn.ModuleList | None = None,
    ) -> torch.Tensor:
        inference_step = self.curr_step

        return self.schedule.forward(
            forward_args, inference_step, transformer_blocks
        )
