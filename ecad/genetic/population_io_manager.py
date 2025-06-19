from abc import ABC, abstractmethod
import dill
import json
from pathlib import Path
import re
from typing import Any

import numpy as np
import numpy.typing as npt

from ecad.schedulers.cache_scheduler.cache_schedule import (
    CacheSchedule,
)
from ecad.types import CacheScheduleDict

# Default base directory for populations
DEFAULT_POPULATIONS_DIR = Path(__file__).parents[2] / Path(
    "results/genetic/populations/"
)
DEFAULT_BENCHMARKS_DIR = Path(__file__).parents[2] / Path(
    "results/benchmark/genetic/populations/"
)


class PopulationIOManager(ABC):
    CONFIG_FILENAME = "manager_config.json"
    CHECKPOINT_FILENAME = "checkpoint.pkl"
    SCORE_KEY = "total_score"
    METRIC_KEY = "total_macs_T"

    def __init__(
        self,
        name: str,
        all_populations_dir: Path = DEFAULT_POPULATIONS_DIR,
        all_benchmarks_dir: Path = DEFAULT_BENCHMARKS_DIR,
        generation_num: int | None = None,
        num_inference_steps: int = 20,
        min_diff_from_default: int = 1,
        population_size: int = 72,
        default_schedule: CacheSchedule | None = None,
        maximize_macs: bool = False,
        candidate_config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Parameters:
            name: Name of the population manager.
            all_populations_dir: Base directory where population schedules will be stored.
            all_benchmarks_dir: Base directory where benchmark scores will be stored.
            generation_num: Generation number to use as current. If None, the greatest existing generation number is used, or 0 if none exist.
            num_inference_steps: Number of inference steps in each schedule.
            min_diff_from_default: Minimum number of differences required from the default schedule.
        """
        self.name = name
        self.population_dir = all_populations_dir / name
        self.population_dir.mkdir(parents=True, exist_ok=True)

        self.benchmark_dir = all_benchmarks_dir / name
        self.benchmark_dir.mkdir(parents=True, exist_ok=True)

        # Determine generation number from existing generation folders if not explicitly provided.
        if generation_num is None:
            existing_gens = []
            for p in self.population_dir.iterdir():
                if p.is_dir() and p.name.startswith("gen_"):
                    try:
                        gen_num = int(p.name.split("_")[1])
                        existing_gens.append(gen_num)
                    except ValueError:
                        continue
            generation_num = max(existing_gens) if existing_gens else 1
        self.generation_num = generation_num

        self.num_inference_steps = num_inference_steps
        self.min_diff_from_default = min_diff_from_default
        self.population_size = population_size

        if default_schedule is None:
            raise ValueError("Default schedule must be provided.")
        self.default_schedule = default_schedule

        self.maximize_macs = maximize_macs
        self.candidate_config: dict[str, Any] = candidate_config or {}

        print(
            f"PopulationIOManager initialized with"
            f"\n\tpopulation_dir: {self.population_dir}"
            f"\n\tbenchmark_dir: {self.benchmark_dir}"
            f"\n\tgeneration_num: {self.generation_num}"
            f"\n\tpopulation_size: {self.population_size}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "population_dir": str(self.population_dir),
            "benchmark_dir": str(self.benchmark_dir),
            "generation_num": self.generation_num,
            "num_inference_steps": self.num_inference_steps,
            "min_diff_from_default": self.min_diff_from_default,
            "population_size": self.population_size,
        }

    def to_json(self, file_path: Path | None = None) -> None:
        """
        Save the PopulationIOManager configuration to a JSON file.
        """
        if file_path is None:
            file_path = self._get_generation_dir() / self.CONFIG_FILENAME
        config = self.to_dict()
        with open(file_path, "w") as f:
            json.dump(config, f, indent=4)

        print(f"Saved PopulationIOManager configuration to {file_path}")

    def save_algorithm(self, algorithm: Any) -> None:
        """
        Save the algorithm checkpoint to generation_dir / self.CHECKPOINT_FILENAME.
        Also, update the manager's generation number from the algorithm state.
        """
        # Sync the generation number from the algorithm if available.
        if hasattr(algorithm, "n_gen") and algorithm.n_gen is not None:
            self.generation_num = algorithm.n_gen

        checkpoint_file = self._get_generation_dir() / self.CHECKPOINT_FILENAME

        with open(checkpoint_file, "wb") as f:
            dill.dump(algorithm, f)
        print(f"Saved algorithm checkpoint to {checkpoint_file}")

    def load_algorithm(self, generation: int | None = None) -> Any:
        """
        Load the algorithm checkpoint from generation_dir / self.CHECKPOINT_FILENAME.
        Updates the manager's generation number based on the loaded algorithm.
        """
        if generation is None:
            generation = self.generation_num
        checkpoint_file = self._get_generation_dir() / self.CHECKPOINT_FILENAME

        with open(checkpoint_file, "rb") as f:
            algorithm = dill.load(f)
        print(
            f"Loaded algorithm checkpoint from {checkpoint_file} with generation {algorithm.n_gen}"
        )
        return algorithm

    def _get_generation_dir(self, generation: int | None = None) -> Path:
        """
        Get the directory for a given generation.
        """
        if generation is None:
            generation = self.generation_num
        gen_dir = self.population_dir / f"gen_{generation:03d}"
        gen_dir.mkdir(parents=True, exist_ok=True)
        return gen_dir

    def _get_candidates_dir(self, generation: int | None = None) -> Path:
        dir = self._get_generation_dir(generation) / "candidates"
        dir.mkdir(parents=True, exist_ok=True)
        return dir

    def _get_candidate_filename(
        self, candidate_index: int, generation: int | None = None
    ) -> Path:
        """
        Get the filename for a candidate within the current generation directory.
        """
        return (
            self._get_candidates_dir(generation)
            / f"cand_{candidate_index:03d}.json"
        )

    def get_pop_gen_dir(self, generation: int | None = None) -> Path:
        return self._get_generation_dir(generation)

    def get_benchmark_gen_dir(self, generation: int | None = None) -> Path:
        if generation is None:
            generation = self.generation_num
        return self.benchmark_dir / f"gen_{generation:03d}/candidates"

    def get_pop_candidates_dir(self, generation: int | None = None) -> Path:
        return self._get_candidates_dir(generation)

    @abstractmethod
    def save_population(
        self,
        population: npt.NDArray,
        generation: int | None = None,
    ) -> None:
        pass

    def ask(
        self,
    ) -> tuple[
        npt.NDArray[np.bool_], npt.NDArray[np.float64], npt.NDArray[np.int64]
    ]:
        X = self.load_population_vectors()
        F = self.load_evaluation_scores()
        G = self.get_constraint_violations(X)
        return X, F, G

    @abstractmethod
    def load_population_schedules(
        self, generation: int | None = None
    ) -> list[tuple[int, CacheSchedule]]:
        pass

    def load_population_vectors(
        self, generation: int | None = None
    ) -> npt.NDArray[np.bool_]:
        pop = self.load_population_schedules(generation)
        result = [schedule.to_numpy(flatten=True) for _, schedule in pop]

        return np.array(result)

    def get_constraint_violations(
        self, population: npt.NDArray[np.bool_]
    ) -> npt.NDArray[np.int64]:
        # TODO: make sure no caching on inital/0th step
        diff_counts = np.sum(
            population != self.default_schedule.to_numpy(flatten=True), axis=1
        )
        G = (self.min_diff_from_default - diff_counts).reshape(-1, 1)

        return G

    def negate_image_reward(
        self,
        scores: dict[int, float],
        max_score: float = 1.0,
    ) -> dict[int, float]:
        for i, score in scores.items():
            scores[i] = max_score - score
        return scores

    def negate_metrics(
        self,
        metrics: dict[int, float],
        max_metric: float = 0.0,
    ) -> dict[int, float]:
        print("WARNING: NEGATING MACS!")
        for i, metric in metrics.items():
            metrics[i] = max_metric - metric
        return metrics

    def load_evaluation_scores(
        self, generation: int | None = None
    ) -> npt.NDArray[np.float64]:
        """
        Load evaluation scores for a given generation.

        Returns:
            A dictionary mapping candidate_index -> {metric_name: x, score_name: y}.
        """
        scores = self._load_scores(generation)
        metrics = self._load_metrics(generation)
        scores = self.negate_image_reward(scores)

        if self.maximize_macs:
            metrics = self.negate_metrics(metrics)

        if scores.keys() != metrics.keys():
            raise ValueError(
                f"WARNING: Candidate indices in scores and metrics do not match. "
                f"Scores: {scores.keys()}. Metrics: {metrics.keys()}"
            )

        results = np.zeros((len(scores), 2))
        for i, score in scores.items():
            results[i, 0] = score
            results[i, 1] = metrics[i]

        return results

    def _load_scores(self, generation: int | None = None) -> dict[int, float]:
        """
        Load evaluation results from a separate directory (scores_dir) for a given generation.

        Assumes each candidate's evaluation results are stored as a JSON file with under a
        directory with the same name that encodes the generation and candidate index.

        Returns:
            A dictionary mapping candidate_index -> ImageReward score, un-normalized.
        """
        scores_dir = self._get_score_dir(generation)
        results = {}

        gen_pattern = re.compile(rf"^cand_(?P<index>\d+)$")
        for dir_path in scores_dir.glob(f"cand_*"):
            if not dir_path.is_dir():
                continue

            jsons = list(dir_path.glob("scores*.json"))

            if len(jsons) == 0:
                print(f"WARNING: No JSON files found in {dir_path}. Skipping.")
                continue

            file_path = jsons[0]
            if len(jsons) > 1:
                print(
                    f"WARNING: Expected 1 JSON file in {dir_path}, found {len(jsons)}. Using {file_path}"
                )

            match = gen_pattern.match(dir_path.name)
            if match is None or "index" not in match.groupdict():
                print(
                    f"WARNING: Could not parse candidate index from {file_path}"
                )
                continue

            index = int(match.group("index"))
            with open(file_path, "r") as f:
                data = json.load(f)

            score = data[self.SCORE_KEY]
            results[index] = score

        return results

    def _load_metrics(self, generation: int | None = None) -> dict[int, float]:
        schedules_dir = self._get_candidates_dir(generation)
        results = {}

        gen_pattern = re.compile(rf"^cand_(?P<index>\d+).json$")

        for file_path in schedules_dir.glob(f"cand_*.json"):
            match = gen_pattern.match(file_path.name)
            if match is None or "index" not in match.groupdict():
                print(
                    f"WARNING: Could not parse candidate index from {file_path}"
                )
                continue

            index = int(match.group("index"))
            with open(file_path, "r") as f:
                data = json.load(f)

            metrics = data["metrics"]

            if self.METRIC_KEY in metrics:
                macs_T = metrics[self.METRIC_KEY]
            # NOTE: this assumes metric key is total_macs_T
            elif "total_macs" in metrics:
                macs_T = metrics["total_macs"] / 1000**4
            else:
                raise ValueError(
                    f"Could not find metric key in metrics for {file_path}"
                )

            results[index] = macs_T

        return results

    def _get_score_dir(self, generation: int | None = None) -> Path:
        if generation is None:
            generation = self.generation_num
        return self.benchmark_dir / f"gen_{generation:03d}/candidates"

    def check_offline_eval(self) -> bool:
        # ensure pop size dirs exist, each with a scores.json file
        scores_dir = self._get_score_dir()
        cand_dirs = [dir for dir in scores_dir.glob("cand_*/") if dir.is_dir()]

        for cand_dir in cand_dirs:
            jsons = list(cand_dir.glob("scores*.json"))
            if len(jsons) < 1:
                return False

        # ensure schedules exist for each candidate, with metrics
        schedules_dir = self._get_candidates_dir()
        cand_jsons = list(schedules_dir.glob("cand_*.json"))

        # don't require it to be equal to self.population size, since we dedup
        if len(cand_dirs) != len(cand_jsons):
            return False

        for file_path in cand_jsons:
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)
                metrics = data["metrics"]
                if (
                    self.METRIC_KEY not in metrics
                    and "total_macs" not in metrics
                ):
                    return False
            except Exception:
                return False

        return True

    @abstractmethod
    def compute_additional_info(
        self,
        schedule: CacheScheduleDict,
    ) -> dict[str, Any]:
        pass

    @staticmethod
    @abstractmethod
    def binary_vector_to_schedule_dict(
        x: np.ndarray, num_inference_steps: int, **kwargs: Any
    ) -> CacheScheduleDict:
        pass
