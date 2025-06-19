import json
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from ecad.genetic.population_io_manager import PopulationIOManager

from ecad.schedulers.cache_scheduler.flux_cache_schedule import (
    FluxCacheSchedule,
)
from ecad.schedulers.cache_scheduler.generators.flux_schedule_generators import (
    gen_default as cache_gen_default,
)
from ecad.types import FluxCacheScheduleDict


# Default base directory for populations
DEFAULT_POPULATIONS_DIR = Path(__file__).parents[2] / Path(
    "results/genetic/flux_populations/"
)
DEFAULT_BENCHMARKS_DIR = Path(__file__).parents[2] / Path(
    "results/benchmark/genetic/flux_populations/"
)


class FluxPopulationIOManager(PopulationIOManager):
    def __init__(
        self,
        name: str,
        all_populations_dir: Path = DEFAULT_POPULATIONS_DIR,
        all_benchmarks_dir: Path = DEFAULT_BENCHMARKS_DIR,
        generation_num: int | None = None,
        num_inference_steps: int = 20,
        min_diff_from_default: int = 1,
        population_size: int = 24,
        num_blocks: int = 19,
        num_single_blocks: int = 38,
        num_component_types_full: int = 3,
        num_component_types_single: int = 3,
        height: int = 256,
        width: int = 256,
        guidance_scale: float = 5,
        maximize_macs = False,
    ) -> None:
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
        super().__init__(
            name=name,
            all_populations_dir=all_populations_dir,
            all_benchmarks_dir=all_benchmarks_dir,
            generation_num=generation_num,
            num_inference_steps=num_inference_steps,
            min_diff_from_default=min_diff_from_default,
            population_size=population_size,
            default_schedule=default_schedule,
            maximize_macs=maximize_macs,
        )

        self.num_blocks: int = num_blocks
        self.num_single_blocks: int = num_single_blocks
        self.num_component_types_single: int = num_component_types_single
        self.num_component_types_full: int = num_component_types_full

        self.height: int = height
        self.width: int = width
        self.guidance_scale: float = guidance_scale

    def to_dict(self) -> dict[str, Any]:
        return super().to_dict() | {
            "num_blocks": self.num_blocks,
            "num_single_blocks": self.num_single_blocks,
            "num_component_types_full": self.num_component_types_full,
            "num_component_types_single": self.num_component_types_single,
        }

    @classmethod
    def from_json(cls, file_path: Path) -> "PopulationIOManager":
        """
        Load a PopulationIOManager configuration from a JSON file.
        """
        with open(file_path, "r") as f:
            config = json.load(f)

        all_population_dir = Path(config["population_dir"]).parent
        all_benchmarks_dir = Path(config["benchmark_dir"]).parent

        args = {
            "name": config["name"],
            "all_populations_dir": all_population_dir,
            "all_benchmarks_dir": all_benchmarks_dir,
        }

        optional_keys = [
            "generation_num",
            "num_inference_steps",
            "min_diff_from_default",
            "population_size",
            "num_blocks",
            "num_single_blocks",
            "num_component_types_full",
            "num_component_types_single",
        ]

        for key in optional_keys:
            if key in config:
                args[key] = config[key]

        return cls(**args)

    def save_population(
        self,
        population: npt.NDArray,
        generation: int | None = None,
    ) -> None:
        """
        Save a population of candidate solutions as JSON files within the current generation folder.

        Parameters:
            population: A 2D NumPy array of shape (n_candidates, n_var) representing candidate decision vectors.
            additional_info: Optional dictionary with additional data to store in each candidate's JSON.
        """
        for i, candidate_vector in enumerate(population):
            schedule_dict = self.binary_vector_to_schedule_dict(
                candidate_vector,
                num_inference_steps=self.num_inference_steps,
                num_blocks=self.num_blocks,
                num_single_blocks=self.num_single_blocks,
                num_component_types_full=self.num_component_types_full,
                num_component_types_single=self.num_component_types_single,
            )
            additional_info = self.compute_additional_info(schedule_dict)
            # Create a CacheSchedule object
            top_level_config = {
                "height": self.height,
                "width": self.width,
                "guidance_scale": self.guidance_scale,
            }

            cache_schedule = FluxCacheSchedule(
                num_blocks=self.num_blocks,
                num_single_blocks=self.num_single_blocks,
                num_inference_steps=self.num_inference_steps,
                name=f"{self.name}_gen_{self.generation_num:03d}_cand_{i:03d}",
                schedule=schedule_dict,
                attributes=additional_info,
                top_level_config=top_level_config,
            )
            candidate_file = self._get_candidate_filename(i, generation)
            cache_schedule.to_json(candidate_file)
            print(
                f"Saved candidate {i} in generation {self.generation_num} to {candidate_file}"
            )

    def load_population_schedules(
        self, generation: int | None = None
    ) -> list[tuple[int, FluxCacheSchedule]]:
        """
        Load candidate schedules from a given generation.

        Returns:
            A list of tuples: (candidate_index, CacheSchedule object)
        """
        candidates = []
        for file_path in self._get_candidates_dir(generation).glob(
            "cand_*.json"
        ):
            parts = file_path.stem.split("_")
            try:
                candidate_index = int(parts[-1])
            except ValueError:
                candidate_index = -1
                print(
                    f"WARNING: Could not parse candidate index from {file_path}"
                )
            cache_schedule = FluxCacheSchedule.from_json(file_path)
            candidates.append((candidate_index, cache_schedule))
        candidates.sort(key=lambda t: t[0])
        return candidates

    def compute_additional_info(
        self,
        schedule: FluxCacheScheduleDict,
    ) -> dict[str, Any]:
        num_affected_steps = 0
        affected_blocks_full = set()
        affected_blocks_single = set()
        total_num_affected_blocks = 0

        for block_schedule, default_block_schedule in zip(
            schedule.values(), self.default_schedule.schedule.values()
        ):
            if block_schedule != default_block_schedule:
                num_affected_steps += 1

            for block_num, block in block_schedule.items():
                default_block = default_block_schedule[block_num]
                if block != default_block:
                    total_num_affected_blocks += 1
                    if block_num.startswith("single_"):
                        affected_blocks_single.add(block_num)
                    else:
                        affected_blocks_full.add(block_num)

        return {
            "num_affected_steps": num_affected_steps,
            "num_affected_blocks": len(affected_blocks_full),
            "num_affected_blocks_single": len(affected_blocks_single),
            "total_num_affected_blocks": total_num_affected_blocks,
        }

    @staticmethod
    def binary_vector_to_schedule_dict(
        x: np.ndarray,
        num_inference_steps: int,
        num_blocks: int,
        num_single_blocks: int,
        num_component_types_full: int,
        num_component_types_single: int,
    ) -> FluxCacheScheduleDict:
        arr = x.reshape(
            (
                num_inference_steps,
                (
                    (num_blocks * num_component_types_full)
                    + (num_single_blocks * num_component_types_single)
                ),
            )
        )
        schedule = {}
        for step in range(num_inference_steps):
            block_schedule = {}
            # full blocks first
            for block_num in range(num_blocks):
                pos = block_num * num_component_types_full

                block_schedule[str(block_num)] = {
                    "full_attn": bool(arr[step, pos + 0]),
                    "full_ff": bool(arr[step, pos + 1]),
                    "full_ff_context": bool(arr[step, pos + 2]),
                }

            # now single blocks
            for block_num in range(num_single_blocks):
                pos = (num_blocks * num_component_types_full) + (
                    block_num * num_component_types_full
                )

                block_schedule[f"single_{block_num}"] = {
                    "single_attn": bool(arr[step, pos + 0]),
                    "single_proj_mlp": bool(arr[step, pos + 1]),
                    "single_proj_out": bool(arr[step, pos + 2]),
                }

            schedule[step] = block_schedule
        return schedule
