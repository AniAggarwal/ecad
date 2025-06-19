from pathlib import Path
import inspect
import sys
from typing import Callable, Iterator

from ecad.schedulers.dit_scheduler.pixart_dit_schedule import PixArtDiTSchedule
from ecad.graph.pixart_builder import PixArtTransformerGraphBuilder

from ecad.schedulers.dit_scheduler.generators.helpers import (
    apply_n_times_centered,
    default_all_timesteps,
    evenly_spaced,
    get_progressive_steps,
    every_other_step,
    default,
    skip_blocks,
    middle_skip,
    middle_repeat,
    parallel,
    reverse,
)
from ecad.utils import PixArtForwardArgs


def gen_skip_block_individual_evenly_spaced(
    num_blocks: int, num_inference_steps: int
) -> Iterator[PixArtDiTSchedule]:
    for num_affected_steps in range(1, num_inference_steps + 1, 2):
        for affected_block in range(num_blocks):
            schedule = default_all_timesteps(num_blocks, num_inference_steps)
            for step in apply_n_times_centered(
                num_inference_steps, num_affected_steps
            ):
                schedule[step] = PixArtTransformerGraphBuilder(skip_blocks(num_blocks, [affected_block]), PixArtForwardArgs)

            name = f"individual_skip_affected_{affected_block:03}_affected_steps_{num_affected_steps:03}"
            attributes = {
                "affected_block": affected_block,
                "num_affected_steps": num_affected_steps,
            }
            yield PixArtDiTSchedule(
                num_blocks,
                num_inference_steps,
                name,
                schedule,
                attributes=attributes,
            )


def gen_default(
    num_blocks: int, num_inference_steps: int
) -> Iterator[PixArtDiTSchedule]:
    config = default(num_blocks)
    schedule_dict = {
        step: PixArtTransformerGraphBuilder(config, PixArtForwardArgs, None)
        for step in range(num_inference_steps)
    }
    yield PixArtDiTSchedule(
        num_blocks, num_inference_steps, "default", schedule_dict
    )


def gen_skip_block_all_timesteps(
    num_blocks: int,
    num_inference_steps: int,
) -> Iterator[PixArtDiTSchedule]:
    for block_to_skip in range(num_blocks):
        schedule = {
            step: PixArtTransformerGraphBuilder(skip_blocks(num_blocks, [block_to_skip]), PixArtForwardArgs)
            for step in range(num_inference_steps)
        }
        name = f"skip_block_{block_to_skip}_all_timesteps"

        yield PixArtDiTSchedule(num_blocks, num_inference_steps, name, schedule)


def gen_skip_block_progressive(
    num_blocks: int,
    num_inference_steps: int,
) -> Iterator[PixArtDiTSchedule]:
    for step_start_skipping in get_progressive_steps(num_inference_steps):
        for block_to_skip in range(num_blocks):
            schedule = {}

            for step in range(num_inference_steps):
                if step < step_start_skipping:
                    schedule[step] = PixArtTransformerGraphBuilder(default(num_blocks), PixArtForwardArgs)
                else:
                    schedule[step] = PixArtTransformerGraphBuilder(skip_blocks(num_blocks, [block_to_skip]), PixArtForwardArgs)

            name = f"skip_block_{block_to_skip}_from_timestep_{step_start_skipping}"
            yield PixArtDiTSchedule(num_blocks, num_inference_steps, name, schedule)


def gen_middle_skip_progressive(
    num_blocks: int,
    num_inference_steps: int,
) -> Iterator[PixArtDiTSchedule]:
    for step_start_skipping in every_other_step(0, num_inference_steps - 1):
        for num_affected_blocks in range(1, num_blocks, 2):
            schedule = {}
            for step in range(num_inference_steps):
                if step < step_start_skipping:
                    schedule[step] = PixArtTransformerGraphBuilder(default(num_blocks), PixArtForwardArgs)
                else:
                    schedule[step] = PixArtTransformerGraphBuilder(middle_skip(num_blocks, num_affected_blocks), PixArtForwardArgs)

            name = f"middle_skip_affected_{num_affected_blocks:03}_from_timestep_{step_start_skipping:03}"
            attributes = {
                "num_affected_blocks": num_affected_blocks,
                "from_timestep": step_start_skipping,
            }
            yield PixArtDiTSchedule(
                num_blocks,
                num_inference_steps,
                name,
                schedule,
                attributes=attributes,
            )


def gen_middle_skip_evenly_spaced(
    num_blocks: int,
    num_inference_steps: int,
) -> Iterator[PixArtDiTSchedule]:
    for num_affected_steps in range(1, num_inference_steps + 1, 1):
        for num_affected_blocks in range(1, num_blocks, 1):
            schedule = default_all_timesteps(num_blocks, num_inference_steps)
            for step in apply_n_times_centered(
                num_inference_steps, num_affected_steps
            ):
                schedule[step] = PixArtTransformerGraphBuilder(middle_skip(num_blocks, num_affected_blocks), PixArtForwardArgs)

            name = f"middle_skip_affected_{num_affected_blocks:03}_affected_steps_{num_affected_steps:03}"
            attributes = {
                "num_affected_blocks": num_affected_blocks,
                "num_affected_steps": num_affected_steps,
            }
            yield PixArtDiTSchedule(
                num_blocks,
                num_inference_steps,
                name,
                schedule,
                attributes=attributes,
            )


def gen_middle_parallel_all_timesteps(
    num_blocks: int,
    num_inference_steps: int,
) -> Iterator[PixArtDiTSchedule]:

    first_parallel = int(num_blocks * 0.25)
    last_parallel = int(num_blocks * 0.75)

    while first_parallel < last_parallel:
        schedule = {
            step: PixArtTransformerGraphBuilder(parallel(
                num_blocks,
                first_parallel,
                last_parallel,
                loop_count=0,
                aggregate_func="avg",
            ), PixArtForwardArgs)
            for step in range(num_inference_steps)
        }
        name = f"middle_parallel_avg_{first_parallel}_to_{last_parallel}_all_timesteps"
        yield PixArtDiTSchedule(num_blocks, num_inference_steps, name, schedule)

        first_parallel += 1
        last_parallel -= 1


def gen_middle_parallel_progressive(
    num_blocks: int,
    num_inference_steps: int,
) -> Iterator[PixArtDiTSchedule]:

    for step_start_skipping in every_other_step(0, num_inference_steps - 1):

        first_parallel = 0
        last_parallel = num_blocks - 1
        while first_parallel < last_parallel:
            # start building schedule with default till step_start_skipping
            # and parallel after with first_parallel to last_parallel
            schedule = {}

            for step in range(num_inference_steps):
                if step < step_start_skipping:
                    schedule[step] = PixArtTransformerGraphBuilder(default(num_blocks), PixArtForwardArgs)
                else:
                    schedule[step] = PixArtTransformerGraphBuilder(parallel(
                        num_blocks,
                        first_parallel,
                        last_parallel,
                        loop_count=0,
                        aggregate_func="avg",
                    ), PixArtForwardArgs)

            num_affected_blocks = last_parallel - first_parallel + 1
            name = f"middle_parallel_avg_affected_{num_affected_blocks:03}_from_timestep_{step_start_skipping:03}"
            attributes = {
                "num_affected_blocks": num_affected_blocks,
                "from_timestep": step_start_skipping,
                "affected_start": first_parallel,
                "affected_end": last_parallel,
            }
            yield PixArtDiTSchedule(
                num_blocks,
                num_inference_steps,
                name,
                schedule,
                attributes=attributes,
            )
            first_parallel += 1
            last_parallel -= 1


def gen_middle_parallel_evenly_spaced(
    num_blocks: int,
    num_inference_steps: int,
) -> Iterator[PixArtDiTSchedule]:

    for num_affected_steps in range(1, num_inference_steps + 1, 2):

        first_parallel = 0
        last_parallel = num_blocks - 1
        while first_parallel < last_parallel:
            schedule = default_all_timesteps(num_blocks, num_inference_steps)

            for step in apply_n_times_centered(
                num_inference_steps, num_affected_steps
            ):
                schedule[step] = PixArtTransformerGraphBuilder(parallel(
                    num_blocks,
                    first_parallel,
                    last_parallel,
                    loop_count=0,
                    aggregate_func="avg",
                ), PixArtForwardArgs)

            num_affected_blocks = last_parallel - first_parallel + 1
            name = f"middle_parallel_avg_affected_{num_affected_blocks:03}_affected_steps_{num_affected_steps:03}"
            attributes = {
                "num_affected_blocks": num_affected_blocks,
                "num_affected_steps": num_affected_steps,
                "affected_start": first_parallel,
                "affected_end": last_parallel,
            }
            yield PixArtDiTSchedule(
                num_blocks,
                num_inference_steps,
                name,
                schedule,
                attributes=attributes,
            )
            first_parallel += 1
            last_parallel -= 1


def gen_middle_looped_parallel_all_timesteps(
    num_blocks: int,
    num_inference_steps: int,
) -> Iterator[PixArtDiTSchedule]:
    for loop_count in range(1, num_blocks):
        first_parallel = int(num_blocks * 0.25)
        last_parallel = int(num_blocks * 0.75)
        while first_parallel < last_parallel:
            schedule = {
                step: PixArtTransformerGraphBuilder(parallel(
                    num_blocks,
                    first_parallel,
                    last_parallel,
                    loop_count=loop_count,
                    aggregate_func="avg",
                ), PixArtForwardArgs)
                for step in range(num_inference_steps)
            }
            name = f"middle_looped_parallel_avg_{first_parallel}_to_{last_parallel}_looped_{loop_count}_all_timesteps"
            yield PixArtDiTSchedule(num_blocks, num_inference_steps, name, schedule)
            first_parallel += 1
            last_parallel -= 1


def gen_middle_looped_parallel_progressive(
    num_blocks: int,
    num_inference_steps: int,
) -> Iterator[PixArtDiTSchedule]:
    for step_start_skipping in [0, 3, 6, 9, 12, 15, 18, 19]:
        for loop_count in range(1, num_blocks):
            first_parallel = 0
            last_parallel = num_blocks - 1
            while first_parallel < last_parallel:
                schedule = {}

                for step in range(num_inference_steps):
                    if step < step_start_skipping:
                        schedule[step] = PixArtTransformerGraphBuilder(default(num_blocks), PixArtForwardArgs)
                    else:
                        schedule[step] = PixArtTransformerGraphBuilder(parallel(
                            num_blocks,
                            first_parallel,
                            last_parallel,
                            loop_count=loop_count,
                            aggregate_func="avg",
                        ), PixArtForwardArgs)

                num_affected_blocks = last_parallel - first_parallel + 1
                name = f"middle_looped_parallel_avg_affected_{num_affected_blocks:03}_looped_{loop_count:03}_from_timestep_{step_start_skipping:03}"
                attributes = {
                    "num_affected_blocks": num_affected_blocks,
                    "from_timestep": step_start_skipping,
                    "affected_start": first_parallel,
                    "affected_end": last_parallel,
                    "loop_count": loop_count,
                }
                yield PixArtDiTSchedule(
                    num_blocks,
                    num_inference_steps,
                    name,
                    schedule,
                    attributes=attributes,
                )
                first_parallel += 2
                last_parallel -= 2


def gen_middle_looped_parallel_evenly_spaced(
    num_blocks: int, num_inference_steps: int
) -> Iterator[PixArtDiTSchedule]:
    # generating a 5 by 5 by 5 grid of schedules
    num_affected_steps_vals = evenly_spaced(1, num_inference_steps, 5)
    loop_count_vals = evenly_spaced(1, num_blocks - 1, 5)

    # one for first_parallel (from 0 upward), one for last_parallel (from num_blocks-1 downward)
    first_parallel_vals = evenly_spaced(0, (num_blocks // 2) - 1, 5)
    last_parallel_vals = evenly_spaced(num_blocks - 1, (num_blocks // 2), 5)

    for num_affected_steps in num_affected_steps_vals:
        for loop_count in loop_count_vals:
            for first_parallel, last_parallel in zip(
                first_parallel_vals, last_parallel_vals
            ):

                print(
                    num_affected_steps,
                    loop_count,
                    first_parallel,
                    last_parallel,
                )

                schedule = default_all_timesteps(
                    num_blocks, num_inference_steps
                )

                for step in apply_n_times_centered(
                    num_inference_steps, num_affected_steps
                ):
                    schedule[step] = PixArtTransformerGraphBuilder(parallel(
                        num_blocks,
                        first_parallel,
                        last_parallel,
                        loop_count=loop_count,
                        aggregate_func="avg",
                    ), PixArtForwardArgs)

                num_affected_blocks = last_parallel - first_parallel + 1
                name = f"middle_looped_parallel_avg_affected_{num_affected_blocks:03}_looped_{loop_count:03}_affected_steps_{num_affected_steps:03}"
                attributes = {
                    "num_affected_blocks": num_affected_blocks,
                    "num_affected_steps": num_affected_steps,
                    "affected_start": first_parallel,
                    "affected_end": last_parallel,
                    "loop_count": loop_count,
                }
                yield PixArtDiTSchedule(
                    num_blocks,
                    num_inference_steps,
                    name,
                    schedule,
                    attributes=attributes,
                )
                first_parallel += 2
                last_parallel -= 2


def gen_middle_repeat_all_timesteps(
    num_blocks: int,
    num_inference_steps: int,
) -> Iterator[PixArtDiTSchedule]:
    assert num_blocks >= 3, "num_blocks must be at least 3 for middle_repeat"

    start_skip = 1
    end_skip = num_blocks - 2

    while start_skip < end_skip:
        schedule = {
            step: PixArtTransformerGraphBuilder(middle_repeat(
                num_blocks,
                start_skip,
                end_skip,
                repeat_block=None,
                repeat_count=None,
            ), PixArtForwardArgs)
            for step in range(num_inference_steps)
        }
        name = f"middle_repeat_{start_skip}_to_{end_skip}_all_timesteps"
        yield PixArtDiTSchedule(num_blocks, num_inference_steps, name, schedule)
        start_skip += 1
        end_skip -= 1


def gen_middle_repeat_progressive(
    num_blocks: int,
    num_inference_steps: int,
) -> Iterator[PixArtDiTSchedule]:
    assert num_blocks >= 3, "num_blocks must be at least 3 for middle_repeat"

    for step_start_skipping in every_other_step(0, num_inference_steps - 1):
        # at least first and last block not skipped
        start_skip = 1
        end_skip = num_blocks - 2

        while start_skip < end_skip:
            schedule = {}

            for step in range(num_inference_steps):
                if step < step_start_skipping:
                    schedule[step] = PixArtTransformerGraphBuilder(default(num_blocks), PixArtForwardArgs)
                else:
                    schedule[step] = PixArtTransformerGraphBuilder(middle_repeat(
                        num_blocks,
                        start_skip,
                        end_skip,
                        repeat_block=None,
                        repeat_count=None,
                    ), PixArtForwardArgs)

            num_affected_blocks = end_skip - start_skip + 1
            name = f"middle_repeat_affected_{num_affected_blocks:03}_from_timestep_{step_start_skipping:03}"
            attributes = {
                "num_affected_blocks": num_affected_blocks,
                "from_timestep": step_start_skipping,
                "affected_start": start_skip,
                "affected_end": end_skip,
            }
            yield PixArtDiTSchedule(
                num_blocks,
                num_inference_steps,
                name,
                schedule,
                attributes=attributes,
            )
            start_skip += 1
            end_skip -= 1


def gen_middle_repeat_evenly_spaced(
    num_blocks: int,
    num_inference_steps: int,
) -> Iterator[PixArtDiTSchedule]:

    for num_affected_steps in range(1, num_inference_steps + 1, 2):
        # at least first and last block not skipped
        start_skip = 1
        end_skip = num_blocks - 2

        while start_skip < end_skip:
            schedule = default_all_timesteps(num_blocks, num_inference_steps)

            for step in apply_n_times_centered(
                num_inference_steps, num_affected_steps
            ):
                schedule[step] = PixArtTransformerGraphBuilder(middle_repeat(
                    num_blocks,
                    start_skip,
                    end_skip,
                    repeat_block=None,
                    repeat_count=None,
                ), PixArtForwardArgs)

            num_affected_blocks = end_skip - start_skip + 1
            name = f"middle_repeat_affected_{num_affected_blocks:03}_affected_steps_{num_affected_steps:03}"
            attributes = {
                "num_affected_blocks": num_affected_blocks,
                "num_affected_steps": num_affected_steps,
                "affected_start": start_skip,
                "affected_end": end_skip,
            }
            yield PixArtDiTSchedule(
                num_blocks,
                num_inference_steps,
                name,
                schedule,
                attributes=attributes,
            )
            start_skip += 1
            end_skip -= 1


def gen_reverse_all_timesteps(
    num_blocks: int, num_inference_steps: int
) -> Iterator[PixArtDiTSchedule]:
    first_to_reverse = 0
    last_to_reverse = num_blocks - 1

    while first_to_reverse < last_to_reverse:
        schedule = {
            step: PixArtTransformerGraphBuilder(reverse(num_blocks, first_to_reverse, last_to_reverse), PixArtForwardArgs)
            for step in range(num_inference_steps)
        }
        name = f"reverse_{first_to_reverse}_to_{last_to_reverse}_all_timesteps"
        yield PixArtDiTSchedule(num_blocks, num_inference_steps, name, schedule)

        first_to_reverse += 2
        last_to_reverse -= 2
        # more granularity on the last step
        if first_to_reverse >= last_to_reverse:
            first_to_reverse -= 1
            last_to_reverse += 1


def gen_middle_reverse_progressive(
    num_blocks: int, num_inference_steps: int
) -> Iterator[PixArtDiTSchedule]:
    for step_start_skipping in every_other_step(0, num_inference_steps - 1):
        first_to_reverse = 0
        last_to_reverse = num_blocks - 1

        while first_to_reverse < last_to_reverse:
            schedule = {}

            for step in range(num_inference_steps):
                if step < step_start_skipping:
                    schedule[step] = PixArtTransformerGraphBuilder(default(num_blocks), PixArtForwardArgs)
                else:
                    schedule[step] = PixArtTransformerGraphBuilder(reverse(num_blocks, first_to_reverse, last_to_reverse),
                                                                   PixArtForwardArgs)

            num_affected_blocks = last_to_reverse - first_to_reverse + 1
            name = f"reverse_num_affected_{num_affected_blocks:03}_from_timestep_{step_start_skipping:03}"
            attributes = {
                "num_affected_blocks": num_affected_blocks,
                "from_timestep": step_start_skipping,
                "affected_start": first_to_reverse,
                "affected_end": last_to_reverse,
            }

            yield PixArtDiTSchedule(
                num_blocks,
                num_inference_steps,
                name,
                schedule,
                attributes=attributes,
            )
            first_to_reverse += 1
            last_to_reverse -= 1


def gen_middle_reverse_evenly_spaced(
    num_blocks: int, num_inference_steps: int
) -> Iterator[PixArtDiTSchedule]:
    for num_affected_steps in range(1, num_inference_steps + 1, 2):
        first_to_reverse = 0
        last_to_reverse = num_blocks - 1

        while first_to_reverse < last_to_reverse:
            schedule = default_all_timesteps(num_blocks, num_inference_steps)
            for step in apply_n_times_centered(
                num_inference_steps, num_affected_steps
            ):
                schedule[step] = PixArtTransformerGraphBuilder(reverse(num_blocks, first_to_reverse, last_to_reverse),
                                                               PixArtForwardArgs)

            num_affected_blocks = last_to_reverse - first_to_reverse + 1
            name = f"reverse_num_affected_{num_affected_blocks:03}_affected_steps_{num_affected_steps:03}"
            attributes = {
                "num_affected_blocks": num_affected_blocks,
                "num_affected_steps": num_affected_steps,
                "affected_start": first_to_reverse,
                "affected_end": last_to_reverse,
            }

            yield PixArtDiTSchedule(
                num_blocks,
                num_inference_steps,
                name,
                schedule,
                attributes=attributes,
            )
            first_to_reverse += 1
            last_to_reverse -= 1


def save_schedules(
    schedules: Iterator[PixArtDiTSchedule],
    output_dir: Path,
    visualize: bool = False,
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

        if visualize:
            schedule.visualize_schedule(output_dir)

    if not at_least_one:
        print("WARNING: No schedules saved.")


# automatically generate the function registry, must be done in a function
# rather than bare module-level code
def get_gen_functions():
    current_module = sys.modules[__name__]
    return {
        name: obj
        for name, obj in inspect.getmembers(current_module, inspect.isfunction)
        if name.startswith("gen_")
    }


GEN_FUNCTIONS: dict[str, Callable[..., Iterator[PixArtDiTSchedule]]] = (
    get_gen_functions()
)
