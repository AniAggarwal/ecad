from typing import Iterable
import numpy as np

from ecad.graph.builder import BuilderConfig
from ecad.graph.pixart_builder import PixArtTransformerGraphBuilder
from ecad.utils import PixArtForwardArgs


def apply_n_times_centered(
    num_inference_steps: int, apply_n_times: int
) -> list[int]:
    pts = np.linspace(
        0, num_inference_steps + 1, num=apply_n_times + 2, endpoint=True
    )[1:-1]
    pts = np.ceil(pts - 1).astype(int).tolist()

    assert len(pts) == apply_n_times
    assert all(0 <= pt < num_inference_steps for pt in pts)

    return pts


def evenly_spaced(start: int, stop: int, count: int) -> list[int]:
    """Return a list of `count` evenly spaced integers from start to stop (inclusive)."""
    if count == 1:
        return [start]
    step = (stop - start) / (count - 1)
    return [int(round(start + i * step)) for i in range(count)]


def get_progressive_steps(num_inference_steps: int) -> list[int]:
    # every other, but always include skipping at the last inference step
    step_starts = list(
        range(int(num_inference_steps * 0.25), num_inference_steps, 2)
    ) + [num_inference_steps - 1]

    return step_starts


def every_other_step(start: int, stop: int) -> list[int]:
    steps = list(range(start, stop, 2))
    # stop is inclusive so add it if it's not there
    if steps[-1] != stop:
        steps.append(stop)
    return steps


def default(num_blocks: int) -> BuilderConfig:
    config = {
        "input": {"outputs": ["0"]},
        "output": {"inputs": [str(num_blocks - 1)]},
    } | {
        str(block): {"inputs": [str(block - 1)], "outputs": [str(block + 1)]}
        for block in range(num_blocks)
    }
    config["0"]["inputs"] = ["input"]
    config[str(num_blocks - 1)]["outputs"] = ["output"]

    return config


def default_all_timesteps(
    num_blocks: int, num_inference_steps: int
) -> dict[int, PixArtTransformerGraphBuilder]:
    return {
        step: PixArtTransformerGraphBuilder(default(num_blocks), PixArtForwardArgs, None)
        for step in range(num_inference_steps)
    }


def skip_blocks(
    num_blocks: int, blocks_to_skip: Iterable[int]
) -> BuilderConfig:
    # blocks_to_skip is inclusive both sides
    config = default(num_blocks)

    for block in blocks_to_skip:
        config[str(block)]["skip"] = True

    return config


def middle_skip(num_blocks: int, num_affected_blocks: int) -> BuilderConfig:
    middle_block = num_blocks // 2
    start_skip = middle_block - (num_affected_blocks // 2)
    end_skip = middle_block + (num_affected_blocks // 2)

    if num_affected_blocks % 2 == 0:
        end_skip -= 1

    config = skip_blocks(num_blocks, range(start_skip, end_skip + 1))

    return config


def middle_repeat(
    num_blocks: int,
    start_skip: int,
    end_skip: int,
    repeat_block: int | None = None,
    repeat_count: int | None = None,
) -> BuilderConfig:
    if repeat_block is None:
        repeat_block = start_skip + ((end_skip - start_skip) // 2)
    if repeat_count is None:
        repeat_count = end_skip - start_skip

    config = skip_blocks(
        num_blocks, [i for i in range(start_skip, end_skip + 1)]
    )

    config[str(repeat_block)]["skip"] = False
    config[str(repeat_block)]["repeat_count"] = repeat_count
    config[str(repeat_block)]["repeat_target"] = str(repeat_block)

    return config


def parallel(
    num_blocks: int,
    first_parallel: int,
    last_parallel: int,
    loop_count: int = 0,
    aggregate_func: str = "add",
) -> BuilderConfig:
    config = default(num_blocks)

    input_to_parallel = (
        str(first_parallel - 1) if first_parallel - 1 >= 0 else "input"
    )
    output_from_parallel = (
        str(last_parallel + 1) if last_parallel + 1 < num_blocks else "output"
    )
    parallel_blocks = [
        str(i) for i in range(first_parallel, last_parallel + 1)
    ]

    # insert dummy nodes before and after the parallel blocks
    config["dummy_before"] = {
        "inputs": [input_to_parallel],
        "outputs": parallel_blocks,
    }
    config["dummy_after"] = {
        "inputs": parallel_blocks,
        "outputs": [output_from_parallel],
        "input_type": aggregate_func,
    }

    config[input_to_parallel]["outputs"] = ["dummy_before"]
    config[output_from_parallel]["inputs"] = ["dummy_after"]

    for block in parallel_blocks:
        config[block]["inputs"] = ["dummy_before"]
        config[block]["outputs"] = ["dummy_after"]

    if loop_count > 0:
        config["dummy_after"]["repeat_count"] = loop_count
        config["dummy_after"]["repeat_target"] = "dummy_before"

    return config


def reverse(
    num_blocks: int,
    first_to_reverse: int,
    last_to_reverse: int,
) -> BuilderConfig:
    config = default(num_blocks)

    # this will cause first block's outputs and last block's inputs to be wrong
    # which is updated after the loop
    for i in range(first_to_reverse, last_to_reverse + 1):
        config[str(i)]["inputs"] = [str(i + 1)]
        config[str(i)]["outputs"] = [str(i - 1)]

    input_to_reverse = (
        str(first_to_reverse - 1) if first_to_reverse - 1 >= 0 else "input"
    )
    output_from_reverse = (
        str(last_to_reverse + 1)
        if last_to_reverse + 1 < num_blocks
        else "output"
    )

    config[input_to_reverse]["outputs"] = [str(last_to_reverse)]
    config[output_from_reverse]["inputs"] = [str(first_to_reverse)]
    config[str(first_to_reverse)]["outputs"] = [output_from_reverse]
    config[str(last_to_reverse)]["inputs"] = [input_to_reverse]

    return config
