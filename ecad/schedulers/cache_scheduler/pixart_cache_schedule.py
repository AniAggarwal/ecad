from ecad.schedulers.cache_scheduler.cache_schedule import (
    CacheSchedule,
)
from ecad.types import CustomFuncDict, PixArtCacheScheduleDict, PixArtComponentScheduleDict
import numpy as np
import numpy.typing as npt


class PixArtCacheSchedule(CacheSchedule[PixArtCacheScheduleDict]):

    @property
    def components(self) -> list[str]:
        return ["attn1", "attn2", "ff"]

    def to_numpy(self, flatten: bool = False) -> npt.NDArray[np.bool_]:
        arr = np.zeros(
            (self.num_inference_steps, self.num_blocks, 3), dtype=np.bool_
        )
        for step, block_schedule in self.schedule.items():
            for block_num, component_schedule in block_schedule.items():
                for i, component in enumerate(self.components):
                    arr[step, int(block_num), i] = component_schedule[
                        component
                    ]
        if flatten:
            arr = arr.flatten()
        return arr

    def get_custom_compute_attn(self, block_num: str) -> CustomFuncDict:
        return self.schedule[self.curr_step][block_num].get(
            "custom_compute_attn", {}
        )  # type: ignore

    def get_custom_compute_ff(self, block_num: str) -> CustomFuncDict:
        return self.schedule[self.curr_step][block_num].get(
            "custom_compute_ff", {}
        )  # type: ignore
