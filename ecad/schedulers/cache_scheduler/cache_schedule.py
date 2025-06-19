from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Generic, TypeVar
import json
import numpy as np

import torch
from numpy import typing as npt

from ecad.types import (
    BlockScheduleDict,
    CacheScheduleDict,
)

CacheScheduleDictT = TypeVar("CacheScheduleDictT", bound=CacheScheduleDict)


class CacheSchedule(ABC, Generic[CacheScheduleDictT]):
    def __init__(
        self,
        num_blocks: int,
        num_inference_steps: int,
        name: str,
        schedule: dict[int | str, BlockScheduleDict] | CacheScheduleDict,
        top_level_config: dict[str, Any] | None = None,
        attributes: dict[str, Any] | None = None,
        metrics: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self.num_inference_steps: int = num_inference_steps
        self.metrics = metrics if metrics is not None else {}
        self.attributes = attributes if attributes is not None else {}
        self.num_blocks: int = num_blocks
        self.name: str = name
        self._last_step = -1

        # ensure inference steps are ints
        self.schedule: CacheScheduleDict = {
            int(step): block_schedule
            for step, block_schedule in schedule.items()
        }
        self.top_level_config = (
            top_level_config if top_level_config is not None else {}
        )
        self.reset_step()

        self._components: list[str] = []

    @property
    @abstractmethod
    def components(self) -> list[str]:
        pass

    @abstractmethod
    def to_numpy(self, flatten: bool = False) -> npt.NDArray[np.bool_]:
        pass

    def reset_step(self) -> None:
        self._last_step = -1

    @property
    def curr_step(self) -> int:
        return self._last_step + 1

    def per_step_callback(self, step: int, timestep: int, **kwargs: Any):
        self._last_step = step

    def get_recompute(self, block_num: str, component: str) -> bool:
        if component not in self.components:
            raise ValueError(
                f"Invalid component {component}. Must be one of {self.components}."
            )
        return self.schedule[self.curr_step][block_num][component]

    def to_dict(self) -> dict[str, Any]:
        return {
            "cache_schedule": {
                "num_blocks": self.num_blocks,
                "num_inference_steps": self.num_inference_steps,
                "name": self.name,
                "attributes": self.attributes,
                "schedule": {
                    f"{step:03}": block_schedule
                    for step, block_schedule in self.schedule.items()
                },
            },
            "config": self.top_level_config,
            "metrics": self.metrics,
        }

    def to_json(self, file_path: Path) -> None:
        """Save the schedule as a JSON file at the specified file path."""
        data = self.to_dict()
        with file_path.open("w") as f:
            json.dump(data, f, indent=4)

    @classmethod
    def from_json(cls, file_path: Path) -> "CacheSchedule":
        """Load the schedule from a JSON file at the specified file path."""

        with file_path.open("r") as f:
            data: dict[str, Any] = json.load(f)

        top_level_config = data.pop("config", None)
        metrics = data.pop("metrics", None)

        class_kwargs = data["cache_schedule"]
        return cls(
            **class_kwargs,
            top_level_config=top_level_config,
            metrics=metrics,
        )
