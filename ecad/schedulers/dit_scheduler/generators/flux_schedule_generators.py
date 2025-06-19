import inspect
import sys
from typing import Callable, Iterator

from ecad.schedulers.dit_scheduler.flux_dit_schedule import (
    FluxDiTSchedule,
)
from ecad.graph.flux_builder import FluxTransformerGraphBuilder
from ecad.utils import FluxForwardArgs


def gen_default(
    num_blocks: int,
    num_single_blocks: int,
    num_inference_steps: int,
    height: int | None = None,
    width: int | None = None,
    guidance_scale: float | None = None,
) -> Iterator[FluxDiTSchedule]:
    schedule_dict = {
        step: FluxTransformerGraphBuilder(
            json_config={},
            forward_args_type=FluxForwardArgs,
            num_blocks=num_blocks,
            num_single_blocks=num_single_blocks,
        )
        for step in range(num_inference_steps)
    }

    if height is None or width is None or guidance_scale is None:
        top_level_config = None
    else:
        top_level_config = {
            "height": height,
            "width": width,
            "guidance_scale": guidance_scale,
        }
    name = f"default_{height}x{width}_gs_{guidance_scale}"
    yield FluxDiTSchedule(
        num_blocks,
        num_inference_steps,
        name,
        schedule_dict,
        top_level_config=top_level_config,
        num_single_blocks=num_single_blocks,
    )


# automatically generate the function registry, must be done in a function
# rather than bare module-level code
def get_gen_functions():
    current_module = sys.modules[__name__]
    return {
        name: obj
        for name, obj in inspect.getmembers(current_module, inspect.isfunction)
        if name.startswith("gen_")
    }


GEN_FUNCTIONS: dict[str, Callable[..., Iterator[FluxDiTSchedule]]] = (
    get_gen_functions()
)
