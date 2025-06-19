from typing import Any
from ecad.schedulers.cache_scheduler.cache_schedule import (
    CacheSchedule,
)
from ecad.types import (
    FluxBlockScheduleDict,
    FluxCacheScheduleDict,
)
import numpy as np
import numpy.typing as npt


class FluxCacheSchedule(CacheSchedule[FluxCacheScheduleDict]):
    def __init__(
        self,
        num_blocks: int,
        num_inference_steps: int,
        name: str,
        schedule: (
            dict[int | str, FluxBlockScheduleDict] | FluxCacheScheduleDict
        ),
        top_level_config: dict[str, Any] | None = None,
        attributes: dict[str, Any] | None = None,
        metrics: dict[str, Any] | None = None,
        num_single_blocks: int = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            num_blocks,
            num_inference_steps,
            name,
            schedule,
            top_level_config,
            attributes,
            metrics,
        )

        if num_single_blocks is None:
            raise ValueError(
                "num_single_blocks must be provided for FluxCacheSchedule"
            )

        self.num_single_blocks: int = num_single_blocks

    def to_dict(self) -> dict[str, Any]:
        data = super().to_dict()
        data["cache_schedule"]["num_single_blocks"] = self.num_single_blocks

        return data

    @property
    def components(self) -> list[str]:
        return [
            "single_attn",
            "single_proj_mlp",
            "single_proj_out",
            "full_attn",
            "full_ff",
            "full_ff_context",
        ]

    def to_numpy(self, flatten: bool = True) -> npt.NDArray[np.bool_]:
        if not flatten:
            raise NotImplementedError(
                "FluxCacheSchedule only supports flatten=True"
            )

        arr = []
        for step, block_schedule in self.schedule.items():
            full_arr = []
            single_arr = []

            for block_num, component_schedule in block_schedule.items():
                if block_num.startswith("single_"):
                    for i, component in enumerate(self.components[:3]):
                        single_arr.append(component_schedule[component])
                else:
                    for i, component in enumerate(self.components[3:]):
                        full_arr.append(component_schedule[component])

            arr.append(full_arr + single_arr)

        arr = np.array(arr, dtype=np.bool_).flatten()

        assert arr.shape[0] == (
            self.num_inference_steps
            * ((self.num_blocks * 3) + (self.num_single_blocks * 3))
        )

        return arr
