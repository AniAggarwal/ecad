import inspect
from pathlib import Path
import sys
from typing import Iterable, Iterator
from ecad.schedulers.cache_scheduler.pixart_cache_schedule import PixArtCacheSchedule

from ecad.schedulers.cache_scheduler.generators.helpers import (
    evenly_spaced_cache,
    middle_cache,
    same_for_all_blocks_one_step,
    default_all_timesteps,
)

from ecad.schedulers.dit_scheduler.generators.helpers import (
    apply_n_times_centered,
)
from ecad.types import ImageGeneratorConfig


def gen_default(
    num_blocks: int,
    num_inference_steps: int,
) -> Iterator[PixArtCacheSchedule]:
    yield PixArtCacheSchedule(
        num_blocks=num_blocks,
        num_inference_steps=num_inference_steps,
        name="default",
        schedule=default_all_timesteps(num_blocks, num_inference_steps),
        attributes={},
    )


def helper_middle_cache_evenly_spaced(
    num_blocks: int,
    num_inference_steps: int,
    attn1: bool,
    attn2: bool,
    ff: bool,
    name_prefix: str,
) -> Iterator[PixArtCacheSchedule]:
    for num_affected_steps in range(1, num_inference_steps + 1, 2):
        for num_affected_blocks in range(1, num_blocks, 2):
            schedule = default_all_timesteps(num_blocks, num_inference_steps)

            for step in apply_n_times_centered(
                num_inference_steps, num_affected_steps
            ):
                schedule[step] = middle_cache(
                    num_blocks, num_affected_blocks, attn1, attn2, ff
                )

            name = f"{name_prefix}_affected_{num_affected_blocks:03}_affected_steps_{num_affected_steps:03}"
            attributes = {
                "num_affected_blocks": num_affected_blocks,
                "num_affected_steps": num_affected_steps,
            }
            yield PixArtCacheSchedule(
                num_blocks, num_inference_steps, name, schedule, attributes
            )


def gen_middle_cache_ca_evenly_spaced(
    num_blocks: int,
    num_inference_steps: int,
) -> Iterator[PixArtCacheSchedule]:
    yield from helper_middle_cache_evenly_spaced(
        num_blocks,
        num_inference_steps,
        True,
        False,
        True,
        "middle_cache_ca_evenly_spaced",
    )


def gen_middle_cache_sa_evenly_spaced(
    num_blocks: int,
    num_inference_steps: int,
) -> Iterator[PixArtCacheSchedule]:
    yield from helper_middle_cache_evenly_spaced(
        num_blocks,
        num_inference_steps,
        False,
        True,
        True,
        "middle_cache_sa_evenly_spaced",
    )


def gen_middle_cache_ff_evenly_spaced(
    num_blocks: int,
    num_inference_steps: int,
) -> Iterator[PixArtCacheSchedule]:
    yield from helper_middle_cache_evenly_spaced(
        num_blocks,
        num_inference_steps,
        True,
        True,
        False,
        "middle_cache_ff_evenly_spaced",
    )


def helper_evenly_cache_evenly_spaced(
    num_blocks: int,
    num_inference_steps: int,
    attn1: bool,
    attn2: bool,
    ff: bool,
    name_prefix: str,
) -> Iterator[PixArtCacheSchedule]:
    for num_affected_steps in range(1, num_inference_steps + 1, 2):
        for num_affected_blocks in range(1, num_blocks, 2):
            schedule = default_all_timesteps(num_blocks, num_inference_steps)

            for step in apply_n_times_centered(
                num_inference_steps, num_affected_steps
            ):
                schedule[step] = evenly_spaced_cache(
                    num_blocks, num_affected_blocks, attn1, attn2, ff
                )

            name = f"{name_prefix}_affected_{num_affected_blocks:03}_affected_steps_{num_affected_steps:03}"
            attributes = {
                "num_affected_blocks": num_affected_blocks,
                "num_affected_steps": num_affected_steps,
            }
            yield PixArtCacheSchedule(
                num_blocks, num_inference_steps, name, schedule, attributes
            )


def gen_evenly_cache_ca_evenly_spaced(
    num_blocks: int,
    num_inference_steps: int,
) -> Iterator[PixArtCacheSchedule]:
    yield from helper_evenly_cache_evenly_spaced(
        num_blocks,
        num_inference_steps,
        True,
        False,
        True,
        "evenly_cache_ca_evenly_spaced",
    )


def gen_evenly_cache_sa_evenly_spaced(
    num_blocks: int,
    num_inference_steps: int,
) -> Iterator[PixArtCacheSchedule]:
    yield from helper_evenly_cache_evenly_spaced(
        num_blocks,
        num_inference_steps,
        False,
        True,
        True,
        "evenly_cache_sa_evenly_spaced",
    )


def gen_evenly_cache_ff_evenly_spaced(
    num_blocks: int,
    num_inference_steps: int,
) -> Iterator[PixArtCacheSchedule]:
    yield from helper_evenly_cache_evenly_spaced(
        num_blocks,
        num_inference_steps,
        True,
        True,
        False,
        "evenly_cache_ff_evenly_spaced",
    )


def helper_recompute_every_n(
    num_blocks: int,
    num_inference_steps: int,
    always_compute_attn1: bool,
    always_compute_attn2: bool,
    always_compute_ff: bool,
    name_prefix: str,
) -> Iterator[PixArtCacheSchedule]:
    """Generates schedules where attn1, attn2, and ff are recomputed every n steps for all blocks.

    Args:
        num_blocks: the number of blocks in the DiT.
        num_inference_steps: the number of inference steps the solver will take.
        always_compute_attn1: whether to always recompute attn1, or to only recompute every n steps.
        always_compute_attn2: whether to always recompute attn2, or to only recompute every n steps.
        always_compute_ff: whether to always recompute the feed forward layer, or to only recompute every n steps.
        name_prefix: the prefix to use for the name of the schedule, e.g. recompute_ca_sa or recompute_all.

    Yields:
        A CacheSchedule object with the schedule corresponding to the parameters.
    """
    for n in range(2, num_inference_steps + 1):
        schedule = {}

        num_affected_blocks = 0
        num_affected_steps = 0
        for i in range(num_inference_steps):
            recompute = i % n == 0
            schedule[i] = same_for_all_blocks_one_step(
                num_blocks,
                recompute or always_compute_attn1,
                recompute or always_compute_attn2,
                recompute or always_compute_ff,
            )
            num_affected_steps += int(recompute)
            if recompute:
                num_affected_blocks = num_blocks

        yield PixArtCacheSchedule(
            num_blocks=num_blocks,
            num_inference_steps=num_inference_steps,
            name=f"{name_prefix}_every_{n:03}",
            schedule=schedule,
            attributes={
                "num_affected_blocks": num_affected_blocks,
                "num_affected_steps": num_affected_steps,
                "recompute_attn1_every": n if not always_compute_attn1 else 1,
                "recompute_attn2_every": n if not always_compute_attn2 else 1,
                "recompute_ff_every": n if not always_compute_ff else 1,
            },
        )


def gen_recompute_all_every_n(
    num_blocks: int,
    num_inference_steps: int,
) -> Iterator[PixArtCacheSchedule]:
    yield from helper_recompute_every_n(
        num_blocks, num_inference_steps, False, False, False, "recompute_all"
    )


def gen_recompute_ca_sa_every_n(
    num_blocks: int,
    num_inference_steps: int,
) -> Iterator[PixArtCacheSchedule]:
    yield from helper_recompute_every_n(
        num_blocks, num_inference_steps, False, False, True, "recompute_ca_sa"
    )


def gen_tgate_without_ca_avg(
    num_blocks: int,
    num_inference_steps: int,
) -> Iterator[PixArtCacheSchedule]:
    assert (
        num_inference_steps == 20
    ), "This TGATE schedule is hardcoded for 20 steps"

    gate_steps = [10, 15]
    sp_intervals = [1, 3, 5]
    fi_intervals = [1]
    warmups = [2]

    yield from helper_tgate_without_ca_avg(
        num_blocks,
        num_inference_steps,
        gate_steps,
        sp_intervals,
        fi_intervals,
        warmups,
    )


def gen_tgate_without_ca_avg_m_k_expanded(
    num_blocks: int,
    num_inference_steps: int,
) -> Iterator[PixArtCacheSchedule]:
    gate_steps = range(1, num_inference_steps + 1, 2)
    sp_intervals = range(1, num_inference_steps, 2)
    fi_intervals = [1]
    warmups = [2]

    yield from helper_tgate_without_ca_avg(
        num_blocks,
        num_inference_steps,
        gate_steps,
        sp_intervals,
        fi_intervals,
        warmups,
    )


def helper_tgate_without_ca_avg(
    num_blocks: int,
    num_inference_steps: int,
    gate_steps: Iterable[int],
    sp_intervals: Iterable[int],
    fi_intervals: Iterable[int],
    warmups: Iterable[int],
) -> Iterator[PixArtCacheSchedule]:
    """Generates schedules corresponding to TGATE, but with direct caching of
    cross attention (attn2 in PixArt), rather than an average of the CA over
    the text embedding and the null embedding.

    Note: TGATE varries 4 parameters:
        - gate_step (called m in the paper)
        - sp_interval (k in the paper)
        - fi_interval (fixed at 1 in the paper)
        - self attention warm up (2 in the paper)

    Before the gate step:
        - self attention is recomputed during warm up; after warm up, recomputed every sp_interval steps.
        - Note: the paper's code does step % sp_interval == 0 after the warm up,
          but the paper' wording implies (step - warmup) % sp_interval == 0.
          This implementation follows the code for consistency.
        - cross attention is always recomputed

    After the gate step:
        - self attention is recomputed every fi_interval steps (same note as above)
        - the cached cross attention is used for the rest of the inference steps
        - in the paper, an average of CA(text), CA(null) is cached; this schedule simply uses CA(text)

    The feed forward layer is always recomputed.

    Args:
        num_blocks: the number of blocks in the DiT.
        num_inference_steps: the number of inference steps the solver will take.

    Yields:
        CacheSchedule: a schedule corresponding to a TGATE schedule with some configuration.
    """

    for gate_step in gate_steps:
        for sp_interval in sp_intervals:
            for fi_interval in fi_intervals:
                for warmup in warmups:
                    num_affected_blocks = 0
                    num_affected_steps = 0
                    schedule = {}

                    for step in range(num_inference_steps):
                        if step < gate_step:  # in sp stage
                            attn1 = (step < warmup) or (
                                step % sp_interval == 0
                            )
                            attn2 = True
                        else:  # in fi stage
                            attn1 = step % fi_interval == 0
                            attn2 = False

                        schedule[step] = same_for_all_blocks_one_step(
                            num_blocks, attn1, attn2, True
                        )

                        # if either was not recomputed, increment counters
                        if not (attn1 and attn2):
                            num_affected_steps += 1
                            num_affected_blocks = num_blocks

                    name = f"tgate_without_ca_avg_m_{gate_step:03}_sp_{sp_interval:03}_fi_{fi_interval:03}_warmup_{warmup:03}"
                    yield PixArtCacheSchedule(
                        num_blocks=num_blocks,
                        num_inference_steps=num_inference_steps,
                        name=name,
                        schedule=schedule,
                        attributes={
                            "num_affected_blocks": num_affected_blocks,
                            "num_affected_steps": num_affected_steps,
                            "gate_step": gate_step,
                            "sp_interval": sp_interval,
                            "fi_interval": fi_interval,
                            "warmup": warmup,
                        },
                    )


def gen_tgate(
    num_blocks: int,
    num_inference_steps: int,
) -> Iterator[PixArtCacheSchedule]:
    assert (
        num_inference_steps == 20
    ), "This TGATE schedule is hardcoded for 20 steps"

    gate_steps = [10, 15]
    sp_intervals = [1, 3, 5]
    fi_intervals = [1]
    warmups = [2]

    yield from helper_tgate(
        num_blocks,
        num_inference_steps,
        gate_steps,
        sp_intervals,
        fi_intervals,
        warmups,
    )


def gen_tgate_1024(
    num_blocks: int,
    num_inference_steps: int,
) -> Iterator[PixArtCacheSchedule]:
    assert (
        num_inference_steps == 20
    ), "This TGATE schedule is hardcoded for 20 steps"

    gate_steps = [9, 10, 11, 14, 15, 16]
    sp_intervals = [1, 3, 5]
    fi_intervals = [1]
    warmups = [2]

    yield from helper_tgate(
        num_blocks,
        num_inference_steps,
        gate_steps,
        sp_intervals,
        fi_intervals,
        warmups,
        "PixArt-alpha/PixArt-XL-2-1024-MS",
    )


def gen_tgate_m_k_expanded(
    num_blocks: int,
    num_inference_steps: int,
) -> Iterator[PixArtCacheSchedule]:
    gate_steps = range(2, num_inference_steps + 1, 2)
    sp_intervals = range(1, num_inference_steps, 2)
    fi_intervals = [1]
    warmups = [2]

    yield from helper_tgate(
        num_blocks,
        num_inference_steps,
        gate_steps,
        sp_intervals,
        fi_intervals,
        warmups,
    )


def helper_tgate(
    num_blocks: int,
    num_inference_steps: int,
    gate_steps: Iterable[int],
    sp_intervals: Iterable[int],
    fi_intervals: Iterable[int],
    warmups: Iterable[int],
    transformer_weights: str | None = None,
) -> Iterator[PixArtCacheSchedule]:
    """Generates schedules corresponding to TGATE (inlcuding average of the CA over the text embedding and the null embedding).

    Note: TGATE varries 4 parameters:
        - gate_step (called m in the paper)
        - sp_interval (k in the paper)
        - fi_interval (fixed at 1 in the paper)
        - self attention warm up (2 in the paper)

    Before the gate step:
        - self attention is recomputed during warm up; after warm up, recomputed every sp_interval steps.
        - Note: the paper's code does step % sp_interval == 0 after the warm up,
          but the paper' wording implies (step - warmup) % sp_interval == 0.
          This implementation follows the code for consistency.
        - cross attention is always recomputed

    After the gate step:
        - self attention is recomputed every fi_interval steps (same note as above)
        - the cached cross attention is used for the rest of the inference steps
        - in the paper, an average of CA(text), CA(null) is cached; this schedule simply uses CA(text)

    The feed forward layer is always recomputed.

    Args:
        num_blocks: the total number of blocks in the DiT.
        num_inference_steps: the total number of inference steps the solver will take.
        gate_steps: the 0-indexed gate steps, when CA hidden state is the averaged result, and CFG stops.
        sp_intervals: a list of self attention recomputation intervals for the sp stage.
        fi_intervals: a list of self attention recomputation intervals for the fi stage.
        warmups: a list of self attention warm up steps.
        transformer_weights: the transformer weights name to use for the schedule.

    Yields:
        a CacheSchedule object with the schedule corresponding to the parameters.
    """

    for gate_step in gate_steps:
        for sp_interval in sp_intervals:
            for fi_interval in fi_intervals:
                for warmup in warmups:
                    num_affected_blocks = 0
                    num_affected_steps = 0
                    schedule = {}

                    for step in range(num_inference_steps):
                        if step < gate_step:  # in sp stage
                            attn1 = (step < warmup) or (
                                step % sp_interval == 0
                            )
                            attn2 = True
                        else:  # in fi stage
                            attn1 = step % fi_interval == 0
                            attn2 = False

                        block_kwargs = {
                            "custom_compute_attn": {
                                "name": "compute_attn_tgate",
                                "kwargs": {
                                    "gate_step": gate_step,
                                },
                            }
                        }
                        schedule[step] = same_for_all_blocks_one_step(
                            num_blocks, attn1, attn2, True, block_kwargs
                        )

                        # if either was not recomputed, increment counters
                        if not (attn1 and attn2):
                            num_affected_steps += 1
                            num_affected_blocks = num_blocks

                    name = f"tgate_m_{gate_step:03}_sp_{sp_interval:03}_fi_{fi_interval:03}_warmup_{warmup:03}"
                    config: ImageGeneratorConfig = {
                        "pipeline": {
                            "name": "tgate",
                            "kwargs": {
                                "gate_step": gate_step,
                            },
                        }
                    }
                    if transformer_weights is not None:
                        config["transformer_weights"] = transformer_weights

                    yield PixArtCacheSchedule(
                        num_blocks=num_blocks,
                        num_inference_steps=num_inference_steps,
                        name=name,
                        schedule=schedule,
                        attributes={
                            "num_affected_blocks": num_affected_blocks,
                            "num_affected_steps": num_affected_steps,
                            "gate_step": gate_step,
                            "sp_interval": sp_interval,
                            "fi_interval": fi_interval,
                            "warmup": warmup,
                        },
                        top_level_config=config,
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
