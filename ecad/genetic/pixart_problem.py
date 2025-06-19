import numpy as np
from pymoo.core.problem import ElementwiseProblem

from ecad.schedulers.cache_scheduler.pixart_cache_schedule import PixArtCacheSchedule
from ecad.schedulers.cache_scheduler.generators.pixart_schedule_generators import (
    gen_default as cache_gen_default,
)


class PixArtCachingScheduleProblem(ElementwiseProblem):
    def __init__(
        self,
        num_inference_steps: int = 20,
        num_blocks: int = 28,
        num_component_types: int = 3,
        min_diff_from_default: int = 1,
        default_schedule: PixArtCacheSchedule | None = None,
        **kwargs,
    ):
        """
        Parameters:
            num_inference_steps (int): Number of inference steps (e.g., 20).
            num_blocks (int): Number of blocks per step (e.g., 28).
            num_component_types (int): Number of component modules per block (e.g., 3 for self-, cross-attention, feed-forward).
            min_diff_from_default (int): Minimum number of differences required from the default schedule.
            default_schedule (PixArtCacheSchedule or None): The default schedule. If None, generated via the default CacheSchedule.
            **kwargs: Additional keyword arguments for the ElementwiseProblem base class.
        """
        self.num_inference_steps: int = num_inference_steps
        self.num_blocks: int = num_blocks
        self.num_attention_types: int = num_component_types
        self.min_diff_from_default: int = min_diff_from_default

        # If no default schedule is provided, construct one
        if default_schedule is None:
            default_schedule = next(cache_gen_default(28, 20))
        self.default_schedule = default_schedule.to_numpy(flatten=True)

        # Total number of decision variables = steps * blocks * attention_types
        self.n_var = num_inference_steps * num_blocks * num_component_types

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
