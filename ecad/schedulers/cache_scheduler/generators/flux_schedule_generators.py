import inspect
from pathlib import Path
import sys
from typing import Iterable, Iterator
from ecad.schedulers.cache_scheduler.flux_cache_schedule import (
    FluxCacheSchedule,
)
from ecad.schedulers.dit_scheduler.generators.helpers import (
    apply_n_times_centered,
    evenly_spaced,
)
from ecad.types import FluxBlockScheduleDict, FluxCacheScheduleDict


def same_for_all_blocks_one_step(
    num_blocks: int,
    num_single_blocks: int,
    single_attn: bool,
    single_proj_mlp: bool,
    single_proj_out: bool,
    full_attn: bool,
    full_ff: bool,
    full_ff_context: bool,
) -> FluxBlockScheduleDict:

    config = {
        str(block_num): {
            "full_attn": full_attn,
            "full_ff": full_ff,
            "full_ff_context": full_ff_context,
        }
        for block_num in range(num_blocks)
    } | {
        f"single_{block_num}": {
            "single_attn": single_attn,
            "single_proj_mlp": single_proj_mlp,
            "single_proj_out": single_proj_out,
        }
        for block_num in range(num_single_blocks)
    }

    return config  # type: ignore


def helper_recompute_every_n(
    num_blocks: int,
    num_single_blocks: int,
    num_inference_steps: int,
    always_single_attn: bool,
    always_single_proj_mlp: bool,
    always_single_proj_out: bool,
    always_full_attn: bool,
    always_full_ff: bool,
    always_full_ff_context: bool,
    name_prefix: str,
) -> Iterator[FluxCacheSchedule]:
    for n in range(2, num_inference_steps + 1):
        schedule = {}

        num_affected_blocks = 0
        num_affected_single_blocks = 0
        num_affected_steps = 0
        for i in range(num_inference_steps):
            recompute = i % n == 0
            schedule[i] = same_for_all_blocks_one_step(
                num_blocks,
                num_single_blocks,
                recompute or always_single_attn,
                recompute or always_single_proj_mlp,
                recompute or always_single_proj_out,
                recompute or always_full_attn,
                recompute or always_full_ff,
                recompute or always_full_ff_context,
            )
            num_affected_steps += int(recompute)
            if recompute:
                num_affected_blocks = num_blocks
                num_affected_single_blocks = num_single_blocks

        yield FluxCacheSchedule(
            num_blocks=num_blocks,
            num_single_blocks=num_single_blocks,
            num_inference_steps=num_inference_steps,
            name=f"{name_prefix}_every_{n:03}",
            schedule=schedule,
            attributes={
                "num_affected_blocks": num_affected_blocks,
                "num_affected_single_blocks": num_affected_single_blocks,
                "num_affected_steps": num_affected_steps,
                "recompute_single_attn_every_n": (
                    n if not always_single_attn else 1
                ),
                "recompute_single_proj_mlp_every_n": (
                    n if not always_single_proj_mlp else 1
                ),
                "recompute_single_proj_out_every_n": (
                    n if not always_single_proj_out else 1
                ),
                "recompute_full_attn_every_n": (
                    n if not always_full_attn else 1
                ),
                "recompute_full_ff_every_n": n if not always_full_ff else 1,
                "recompute_full_ff_context_every_n": (
                    n if not always_full_ff_context else 1
                ),
            },
        )


def helper_evenly_cache_evenly_spaced(
    num_blocks: int,
    num_single_blocks: int,
    num_inference_steps: int,
    always_single_attn: bool,
    always_single_proj_mlp: bool,
    always_single_proj_out: bool,
    always_full_attn: bool,
    always_full_ff: bool,
    always_full_ff_context: bool,
    name_prefix: str,
    every_s_steps: int = 3,
    every_b_blocks: int = 3,
) -> Iterator[FluxCacheSchedule]:
    for num_affected_steps in range(1, num_inference_steps + 1, every_s_steps):
        for num_affected_blocks in range(1, num_blocks + num_single_blocks, every_b_blocks):
            schedule = default_all_timesteps(
                num_blocks, num_single_blocks, num_inference_steps
            )
            for step in apply_n_times_centered(
                num_inference_steps, num_affected_steps
            ):
                schedule[step] = evenly_spaced_cache_flux(
                    num_blocks,
                    num_single_blocks,
                    num_affected_blocks,
                    always_single_attn,
                    always_single_proj_mlp,
                    always_single_proj_out,
                    always_full_attn,
                    always_full_ff,
                    always_full_ff_context,
                )

            yield FluxCacheSchedule(
                num_blocks=num_blocks,
                num_single_blocks=num_single_blocks,
                num_inference_steps=num_inference_steps,
                name=f"{name_prefix}_affected_{num_affected_blocks:03}_affected_steps_{num_affected_steps:03}",
                schedule=schedule,
                attributes={
                    "num_total_affected_blocks": num_affected_blocks,
                    "num_affected_steps": num_affected_steps,
                },
            )


def default_one_step(
    num_blocks: int, num_single_blocks: int
) -> FluxBlockScheduleDict:
    return same_for_all_blocks_one_step(
        num_blocks, num_single_blocks, True, True, True, True, True, True
    )


def default_all_timesteps(
    num_blocks: int, num_single_blocks: int, num_inference_steps: int
) -> FluxCacheScheduleDict:
    return {
        i: default_one_step(num_blocks, num_single_blocks)
        for i in range(num_inference_steps)
    }


def evenly_spaced_cache_flux(
    num_blocks: int,
    num_single_blocks: int,
    num_affected_blocks: int,
    single_attn: bool,
    single_proj_mlp: bool,
    single_proj_out: bool,
    full_attn: bool,
    full_ff: bool,
    full_ff_context: bool,
) -> FluxBlockScheduleDict:
    cache_blocks = evenly_spaced(
        0, num_blocks + num_single_blocks - 1, num_affected_blocks
    )
    config = default_one_step(num_blocks, num_single_blocks)

    for block in cache_blocks:
        if block < num_blocks:
            config[str(block)] = {
                "full_attn": full_attn,
                "full_ff": full_ff,
                "full_ff_context": full_ff_context,
            }
        else:
            config[f"single_{block - num_blocks}"] = {
                "single_attn": single_attn,
                "single_proj_mlp": single_proj_mlp,
                "single_proj_out": single_proj_out,
            }

    return config


def gen_default(
    num_blocks: int,
    num_single_blocks: int,
    num_inference_steps: int,
    height: int | None = None,
    width: int | None = None,
    guidance_scale: float | None = None,
) -> Iterator[FluxCacheSchedule]:
    if height is None or width is None or guidance_scale is None:
        top_level_config = None
    else:
        top_level_config = {
            "height": height,
            "width": width,
            "guidance_scale": guidance_scale,
        }
    name = f"default_{height}x{width}_gs_{guidance_scale}"
    yield FluxCacheSchedule(
        num_blocks=num_blocks,
        num_inference_steps=num_inference_steps,
        name=name,
        num_single_blocks=num_single_blocks,
        schedule=default_all_timesteps(
            num_blocks, num_single_blocks, num_inference_steps
        ),
        top_level_config=top_level_config,
        attributes={},
    )


def gen_default_256(
    num_blocks: int,
    num_single_blocks: int,
    num_inference_steps: int,
) -> Iterator[FluxCacheSchedule]:
    yield from gen_default(
        num_blocks,
        num_single_blocks,
        num_inference_steps,
        height=256,
        width=256,
    )


def gen_default_1024(
    num_blocks: int,
    num_single_blocks: int,
    num_inference_steps: int,
) -> Iterator[FluxCacheSchedule]:
    yield from gen_default(
        num_blocks,
        num_single_blocks,
        num_inference_steps,
        height=1024,
        width=1024,
    )


def gen_default_varied_guidance_256(
    num_blocks: int,
    num_single_blocks: int,
    num_inference_steps: int,
) -> Iterator[FluxCacheSchedule]:
    guidance_scales = [2, 3.5, 5, 7]

    for guidance_scale in guidance_scales:
        yield from gen_default(
            num_blocks,
            num_single_blocks,
            num_inference_steps,
            height=256,
            width=256,
            guidance_scale=guidance_scale,
        )


def gen_recompute_all_every_n(
    num_blocks: int,
    num_single_blocks: int,
    num_inference_steps: int,
) -> Iterator[FluxCacheSchedule]:
    yield from helper_recompute_every_n(
        num_blocks,
        num_single_blocks,
        num_inference_steps,
        False,
        False,
        False,
        False,
        False,
        False,
        "recompute_all",
    )


def gen_recompute_attn_every_n(
    num_blocks: int,
    num_single_blocks: int,
    num_inference_steps: int,
) -> Iterator[FluxCacheSchedule]:
    yield from helper_recompute_every_n(
        num_blocks,
        num_single_blocks,
        num_inference_steps,
        False,
        True,
        True,
        False,
        True,
        True,
        "recompute_attn",
    )


def gen_evenly_cache_mlp_ff_evenly_spaced(
    num_blocks: int,
    num_single_blocks: int,
    num_inference_steps: int,
) -> Iterator[FluxCacheSchedule]:
    yield from helper_evenly_cache_evenly_spaced(
        num_blocks,
        num_single_blocks,
        num_inference_steps,
        True,
        False,
        False,
        True,
        False,
        False,
        "evenly_cache_mlp_ff_evenly_spaced",
        3,
        5,
    )

def gen_evenly_cache_single_full_attn_evenly_spaced(
    num_blocks: int,
    num_single_blocks: int,
    num_inference_steps: int,
) -> Iterator[FluxCacheSchedule]:
    yield from helper_evenly_cache_evenly_spaced(
        num_blocks,
        num_single_blocks,
        num_inference_steps,
        False,
        True,
        True,
        False,
        True,
        True,
        "evenly_cache_mlp_ff_evenly_spaced",
        5,
        15,
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


GEN_FUNCTIONS = get_gen_functions()
