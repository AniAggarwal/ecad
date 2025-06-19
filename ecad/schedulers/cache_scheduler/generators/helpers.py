from pathlib import Path
from typing import Iterator
from ecad.schedulers.cache_scheduler.cache_schedule import CacheSchedule
from ecad.schedulers.dit_scheduler.generators.helpers import (
    evenly_spaced,
)
from ecad.types import (
    CustomFuncDict,
    PixArtBlockScheduleDict,
    PixArtCacheScheduleDict,
)

def save_schedules(
    schedules: Iterator[CacheSchedule],
    output_dir: Path,
    skip_existing: bool = True,
) -> None:
    at_least_one = False
    print(f"Saving schedules to {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    for schedule in schedules:
        schedule_fname = output_dir / f"{schedule.name}.json"

        if skip_existing and schedule_fname.exists():
            continue

        while schedule_fname.exists():
            print(f"File {schedule_fname} already exists, incrementing name")
            schedule_fname = schedule_fname.with_name(
                f"{schedule_fname.stem}_1{schedule_fname.suffix}"
            )

        schedule.to_json(schedule_fname)
        print(f"Saved schedule {schedule.name} to {schedule_fname}.")
        at_least_one = True

    if not at_least_one:
        print("WARNING: No schedules saved.")

def same_for_all_blocks_one_step(
    num_blocks: int,
    attn1: bool,
    attn2: bool,
    ff: bool,
    block_kwargs: dict[str, CustomFuncDict] | None = None,
) -> PixArtBlockScheduleDict:
    block_kwargs = {} if block_kwargs is None else block_kwargs
    config = {
        str(block_num): (
            {
                "attn1": attn1,
                "attn2": attn2,
                "ff": ff,
            }
            | block_kwargs
        )
        for block_num in range(num_blocks)
    }
    return config  # type: ignore


def default_one_step(num_blocks: int) -> PixArtBlockScheduleDict:
    return same_for_all_blocks_one_step(num_blocks, True, True, True)


def default_all_timesteps(
    num_blocks: int, num_inference_steps: int
) -> PixArtCacheScheduleDict:
    return {
        i: default_one_step(num_blocks) for i in range(num_inference_steps)
    }


def middle_cache(
    num_blocks: int,
    num_affected_blocks: int,
    attn1: bool,
    attn2: bool,
    ff: bool,
) -> PixArtBlockScheduleDict:
    middle_block = num_blocks // 2
    start = middle_block - (num_affected_blocks // 2)
    end = middle_block + (num_affected_blocks // 2)

    if num_affected_blocks % 2 == 0:
        end -= 1

    cache_blocks = range(start, end + 1)
    config = default_one_step(num_blocks)

    for block in cache_blocks:
        config[str(block)] = {
            "attn1": attn1,
            "attn2": attn2,
            "ff": ff,
        }

    return config


def evenly_spaced_cache(
    num_blocks: int,
    num_affected_blocks: int,
    attn1: bool,
    attn2: bool,
    ff: bool,
) -> PixArtBlockScheduleDict:
    """Creates a block schedule for one inference step where every `num_affected_blocks` block is affected.

    The affected blocks are evenly spaced out across the cache. If a particular component
    is set to False, then it will be NOT be recomputed, and its cached value will be used
    on this inference step.

    Args:
        num_blocks: the total number of blocks in the transformer.
        num_affected_blocks: the number of blocks to apply the cache schedule to.
        attn1: set to True to recompute the first attention layer, self-attention.
        attn2: set to True to recompute the second attention layer, cross-attention.
        ff: set to True to recompute the feed-forward layer.

    Returns:
        A dictionary where the keys are the block numbers and the values are ComponentScheduleDicts.
    """
    cache_blocks = evenly_spaced(0, num_blocks - 1, num_affected_blocks)
    config = default_one_step(num_blocks)

    for block in cache_blocks:
        config[str(block)] = {
            "attn1": attn1,
            "attn2": attn2,
            "ff": ff,
        }

    return config
