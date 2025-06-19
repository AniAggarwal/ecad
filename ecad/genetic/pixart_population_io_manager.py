import dill
import json
from pathlib import Path
import re
from typing import Any

import numpy as np
import numpy.typing as npt

from ecad.genetic.population_io_manager import PopulationIOManager
from ecad.schedulers.cache_scheduler.pixart_cache_schedule import (
    PixArtCacheSchedule,
)
from ecad.schedulers.cache_scheduler.generators.pixart_schedule_generators import (
    gen_default as cache_gen_default,
)
from ecad.types import PixArtCacheScheduleDict

# Default base directory for populations
DEFAULT_POPULATIONS_DIR = Path(__file__).parents[2] / Path(
    "results/genetic/populations/"
)
DEFAULT_BENCHMARKS_DIR = Path(__file__).parents[2] / Path(
    "results/benchmark/genetic/populations/"
)


class PixArtPopulationIOManager(PopulationIOManager):
    def __init__(
        self,
        name: str,
        all_populations_dir: Path = DEFAULT_POPULATIONS_DIR,
        all_benchmarks_dir: Path = DEFAULT_BENCHMARKS_DIR,
        generation_num: int | None = None,
        num_inference_steps: int = 20,
        min_diff_from_default: int = 1,
        population_size: int = 72,
        num_blocks: int = 28,
        num_component_types: int = 3,
        maximize_macs: bool = False,
    ) -> None:
        default_schedule = next(
            cache_gen_default(num_blocks, num_inference_steps)
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
        self.num_blocks = num_blocks
        self.num_component_types = num_component_types

    def to_dict(self) -> dict[str, Any]:
        return super().to_dict() | {
            "num_blocks": self.num_blocks,
            "num_component_types": self.num_component_types,
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
            "num_blocks",
            "num_component_types",
            "min_diff_from_default",
            "population_size",
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
                num_component_types=self.num_component_types,
            )
            additional_info = self.compute_additional_info(schedule_dict)
            # Create a CacheSchedule object.
            cache_schedule = PixArtCacheSchedule(
                num_blocks=self.num_blocks,
                num_inference_steps=self.num_inference_steps,
                name=f"{self.name}_gen_{self.generation_num:03d}_cand_{i:03d}",
                schedule=schedule_dict,
                attributes=additional_info,
                top_level_config=self.candidate_config,
            )
            candidate_file = self._get_candidate_filename(i, generation)
            cache_schedule.to_json(candidate_file)
            print(
                f"Saved candidate {i} in generation {self.generation_num} to {candidate_file}"
            )

    def load_population_schedules(
        self, generation: int | None = None
    ) -> list[tuple[int, PixArtCacheSchedule]]:
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
            cache_schedule = PixArtCacheSchedule.from_json(file_path)
            schedule_config = cache_schedule.top_level_config

            if not self.candidate_config:
                self.candidate_config = schedule_config
            else:
                if self.candidate_config != schedule_config:
                    print(
                        f"WARNING: Candidate config mismatch. Expected {self.candidate_config}, got {schedule_config}"
                    )

            candidates.append((candidate_index, cache_schedule))
        candidates.sort(key=lambda t: t[0])
        return candidates

    def load_population_vectors(
        self, generation: int | None = None
    ) -> npt.NDArray[np.bool_]:
        pop = self.load_population_schedules(generation)
        result = [schedule.to_numpy(flatten=True) for _, schedule in pop]

        return np.array(result)

    def get_constraint_violations(
        self, population: npt.NDArray[np.bool_]
    ) -> npt.NDArray[np.int64]:
        diff_counts = np.sum(
            population != self.default_schedule.to_numpy(flatten=True), axis=1
        )
        G = (self.min_diff_from_default - diff_counts).reshape(-1, 1)

        return G

    def compute_additional_info(
        self,
        schedule: PixArtCacheScheduleDict,
    ) -> dict[str, Any]:

        num_affected_steps = 0
        affected_blocks = set()
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
                    affected_blocks.add(block_num)

        return {
            "num_affected_steps": num_affected_steps,
            "num_affected_blocks": len(affected_blocks),
            "total_num_affected_blocks": total_num_affected_blocks,
        }

    @staticmethod
    def binary_vector_to_schedule_dict(
        x: np.ndarray,
        num_inference_steps: int,
        num_blocks: int,
        num_component_types: int,
    ) -> PixArtCacheScheduleDict:
        """
        Convert a binary decision vector x into a schedule dictionary.

        The input vector x is expected to have length:
            num_inference_steps * num_blocks * num_component_types.

        It reshapes x into a (num_inference_steps, num_blocks, num_component_types)
        boolean array and maps each component to keys "attn1", "attn2", and "ff".
        """
        arr = x.reshape((num_inference_steps, num_blocks, num_component_types))
        schedule = {}
        for step in range(num_inference_steps):
            block_schedule = {}
            for block_num in range(num_blocks):
                block_schedule[str(block_num)] = {
                    "attn1": bool(arr[step, block_num, 0]),
                    "attn2": bool(arr[step, block_num, 1]),
                    "ff": bool(arr[step, block_num, 2]),
                }
            schedule[step] = block_schedule
        return schedule
