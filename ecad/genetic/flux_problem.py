from pymoo.core.problem import ElementwiseProblem
from ecad.schedulers.cache_scheduler.flux_cache_schedule import (
    FluxCacheSchedule,
)
import numpy as np

from ecad.schedulers.cache_scheduler.generators.flux_schedule_generators import (
    gen_default as cache_gen_default,
)


class FluxCachingScheduleProblem(ElementwiseProblem):
    def __init__(
        self,
        num_inference_steps: int = 20,
        num_blocks: int = 19,
        num_single_blocks: int = 38,
        num_component_types_full: int = 3,
        num_component_types_single: int = 3,
        min_diff_from_default: int = 1,
        default_schedule: FluxCacheSchedule | None = None,
        height: int = 256,
        width: int = 256,
        guidance_scale: float = 5,
        **kwargs,
    ):
        self.num_inference_steps: int = num_inference_steps
        self.num_blocks: int = num_blocks
        self.num_single_blocks: int = num_single_blocks
        self.num_component_types_single: int = num_component_types_single
        self.num_component_types_full: int = num_component_types_full
        self.min_diff_from_default: int = min_diff_from_default
        self.height: int = height
        self.width: int = width
        self.guidance_scale: float = guidance_scale

        # If no default schedule is provided, construct one
        if default_schedule is None:
            default_schedule = next(
                cache_gen_default(
                    num_blocks,
                    num_single_blocks,
                    num_inference_steps,
                    height,
                    width,
                    guidance_scale,
                )
            )
        self.default_schedule = default_schedule.to_numpy(flatten=True)

        # Total number of decision variables = steps * blocks * attention_types
        self.n_var = num_inference_steps * (
            (num_blocks * num_component_types_full)
            + (num_single_blocks * num_component_types_single)
        )

        if self.default_schedule.size != self.n_var:
            raise ValueError(
                f"Default schedule shape's size {self.default_schedule.shape} -> {self.default_schedule.size} does not match n_var {self.n_var}."
            )

        # Decision variables are binary: lower bound 0 and upper bound 1.
        xl = np.zeros(self.n_var)
        xu = np.ones(self.n_var)

        # Two objectives:
        # 1. Minimize MACs (we expect more caching to reduce MACs)
        # 2. Maximize ImageReward -> minimize negative ImageReward
        super().__init__(
            n_var=self.n_var,
            n_obj=2,
            n_ieq_constr=1,
            xl=xl,
            xu=xu,
            vtype=np.bool_,
            **kwargs,
        )

    def _evaluate(self, x, out, *args, **kwargs):
        raise NotImplementedError(
            "Evaluation function not implemented, since we use the ask/tell API."
        )
