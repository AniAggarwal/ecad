"""
Base functionality for NSGA-II training with offline evaluation.
This module contains common code shared between single GPU and SLURM implementations.
"""

import argparse
from typing import Any, Optional
from pathlib import Path

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.termination import NoTermination
from pymoo.core.population import Population
from pymoo.operators.crossover.pntx import PointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation

from ecad.genetic.flux_population_io_manager import (
    FluxPopulationIOManager,
)
from ecad.genetic.flux_problem import FluxCachingScheduleProblem
from ecad.genetic.pixart_population_io_manager import (
    PixArtPopulationIOManager,
)
from ecad.genetic.pixart_problem import PixArtCachingScheduleProblem
from ecad.genetic.sampling import BinaryRandomSampling
from ecad.genetic.population_io_manager import (
    PopulationIOManager,
    DEFAULT_POPULATIONS_DIR,
    DEFAULT_BENCHMARKS_DIR,
)
from ecad.image_generators.image_generator import ImageGenerator
from ecad.image_generators.load_image_generator import (
    ImageGeneratorRegistry,
    get_image_generator_type,
)
from ecad.image_generators.pixart_image_generator import (
    PixArtImageGenerator,
)
from ecad.image_generators.flux_image_generator import (
    FluxImageGenerator,
)


def get_base_argparser() -> argparse.ArgumentParser:
    """
    Get the base argument parser with common arguments for both single GPU and SLURM.
    """
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument(
        "--image-generator",
        type=str,
        required=True,
        help="The name of the image generator to use.",
        choices=list(ImageGeneratorRegistry.registry.keys()),
    )
    parser.add_argument(
        "--load-from",
        type=Path,
        help="Load an existing population manager from a directory. If not provided, all other arguments are required.",
    )
    parser.add_argument(
        "--name", type=str, help="Name of the population manager"
    )
    parser.add_argument(
        "--num-cycles",
        default="1",
        type=str,
        help="Number of cycles to run (default: 1). If `inf` is provided, run indefinitely.",
    )
    parser.add_argument(
        "--print-not-submit",
        action="store_true",
        help="Print commands to run offline evaluation scripts without submitting them. Num cycles must be 1.",
    )
    parser.add_argument(
        "--all-populations-dir",
        type=Path,
        default=DEFAULT_POPULATIONS_DIR,
        help="Base directory for population schedules",
    )
    parser.add_argument(
        "--all-benchmarks-dir",
        type=Path,
        default=DEFAULT_BENCHMARKS_DIR,
        help="Base directory for benchmark scores",
    )
    parser.add_argument(
        "--embedding-dir",
        type=Path,
        required=True,
        help="The directory containing embeddings used to generate images",
    )
    parser.add_argument(
        "--generation-num",
        type=int,
        default=None,
        help="Generation number to start from (default: use highest existing or 0)",
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=20,
        help="Number of inference steps",
    )
    parser.add_argument(
        "--benchmark-prompts",
        type=Path,
        required=True,
        help="JSON file containing benchmark prompts.",
    )
    parser.add_argument(
        "--file-mode",
        choices=["json", "text", "txt", "tsv"],
        help="For scoring images. JSON is the Image Reward prompts mode. TSV must have a column named 'Prompt', and have a header line. Text format is a simple text file with a prompt on each line.",
    )
    parser.add_argument(
        "--image-naming-mode",
        choices=["image_reward", "parti"],
        help="The pattern to use to match image filenames. Use image_reward for image_reward embeddings, parti for parti and drawbench embeddings.",
    )
    parser.add_argument(
        "--min-diff-from-default",
        type=int,
        default=1,
        help="Minimum number of differences required from default schedule",
    )
    parser.add_argument(
        "--population-size",
        type=int,
        default=72,
        help="Population size for NSGA-II",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for image generation runs",
    )
    parser.add_argument(
        "--num-images-per-prompt",
        type=int,
        default=10,
        help="Number of images to generate per prompt.",
    )
    parser.add_argument(
        "--exactly-n-images",
        type=int,
        default=1000,
        help="Number of images to expect before scoring them",
    )
    parser.add_argument(
        "--regen-if-not-n-images",
        type=int,
        default=1000,
        help="Force re-generating images for schedules whose directories do not contain exactly n images.",
    )
    parser.add_argument(
        "--maximize-MACs",
        action="store_true",
        help="Maximize MACs instead of minimizing them.",
    )

    return parser


def validate_base_args(args: argparse.Namespace) -> None:
    """
    Validate common arguments.
    """
    if args.load_from is None and args.name is None:
        raise ValueError("Must provide a name or path to JSON file.")

    if args.num_cycles.isdigit():
        args.num_cycles = int(args.num_cycles)
    elif args.num_cycles != "inf":
        raise ValueError("num_cycles must be an integer or 'inf'.")

    if args.num_cycles != 1 and args.print_not_submit:
        raise ValueError(
            "print-not-submit can only be used with num_cycles=1."
        )


def init_gen_0(
    manager: PopulationIOManager, image_generator_type: type[ImageGenerator]
) -> NSGA2:
    """
    Initialize generation 0 of the NSGA-II algorithm.
    """
    print(
        "Attempting to load seed population. Should be under gen_000/candidates/cand_XXX.json"
    )
    sample = False
    X = manager.load_population_vectors(0)

    if X.size == 0:
        if (
            input(
                "No initial population found. Randomly seed the population? (y/N) "
            )
            == "y"
        ):
            sample = True
        else:
            raise ValueError("No initial population found")

    print("Initializing new algorithm.")
    if issubclass(image_generator_type, PixArtImageGenerator):
        assert isinstance(manager, PixArtPopulationIOManager)
        problem = PixArtCachingScheduleProblem(
            manager.num_inference_steps,
            manager.num_blocks,
            manager.num_component_types,
            manager.min_diff_from_default,
        )
    elif issubclass(image_generator_type, FluxImageGenerator):
        assert isinstance(manager, FluxPopulationIOManager)
        problem = FluxCachingScheduleProblem(
            manager.num_inference_steps,
            manager.num_blocks,
            manager.num_single_blocks,
            manager.num_component_types_full,
            manager.num_component_types_single,
            manager.min_diff_from_default,
        )
    else:
        raise ValueError("Unsupported image generator type.")

    if sample:
        sampling_or_pop = BinaryRandomSampling()
    else:
        sampling_or_pop = Population.new("X", X)

    algorithm = NSGA2(
        pop_size=manager.population_size,
        sampling=sampling_or_pop,
        crossover=PointCrossover(prob=0.9, n_points=4),
        mutation=BitflipMutation(prob=0.05),
    )
    print("Setting up algorithm.")
    termination = NoTermination()
    algorithm.setup(problem, termination=termination, seed=0, verbose=True)
    init_pop = algorithm.ask()

    # need to sync, since algo now has gen 1, while manager has gen 0 for seed gen
    manager.generation_num = algorithm.n_gen
    X = init_pop.get("X")

    manager.save_population(X, 1)
    print(f"Candidates for generation {manager.generation_num} saved.")

    return algorithm


def train_one_cycle(manager: PopulationIOManager, algorithm: NSGA2) -> bool:
    """
    Run one cycle of the NSGA-II training.
    Returns True if offline evaluation was found and processed, False otherwise.
    """
    if not manager.check_offline_eval():
        print(
            f"Offline evaluation not found for generation {manager.generation_num}."
        )
        return False
    else:
        # Offline evaluation is complete; load evaluation results.
        print("Offline evaluation detected. Loading evaluation results...")
        X, F, G = manager.ask()
        sampling_or_pop = Population.new("X", X, F=F, G=G)
        algorithm.tell(sampling_or_pop)

        # Sync generation number between algorithm and IO manager.
        manager.generation_num = algorithm.n_gen

        sampling_or_pop = algorithm.ask()
        X = sampling_or_pop.get("X")
        manager.save_population(X)
        print(
            f"Candidates for generation {manager.generation_num} saved. "
            "Please run your offline evaluation scripts and then re-run this script to resume. "
            "Commands to run the offline evaluation are printed below."
        )
        return True


def checkpoint_manager_and_algorithm(
    manager: PopulationIOManager, algorithm: NSGA2
) -> None:
    """
    Save checkpoint of the manager and algorithm state.
    """
    manager.save_algorithm(algorithm)
    manager.to_json()
    print(
        f"Checkpointed manager config and algorithm for generation {manager.generation_num}"
    )


def initialize_manager(
    args: argparse.Namespace, image_generator_type: type[ImageGenerator]
) -> PopulationIOManager:
    """
    Initialize the population IO manager based on the image generator type.
    """
    if issubclass(image_generator_type, PixArtImageGenerator):
        manager_type = PixArtPopulationIOManager
    elif issubclass(image_generator_type, FluxImageGenerator):
        manager_type = FluxPopulationIOManager
    else:
        raise ValueError(
            f"Unsupported Image Generator type {image_generator_type}."
        )

    if args.load_from is not None:
        if not args.load_from.exists():
            raise ValueError(f"JSON file {args.load_from} does not exist.")
        manager = manager_type.from_json(args.load_from)
    else:
        # Initialize the IO manager with the provided arguments.
        manager = manager_type(
            name=args.name,
            all_populations_dir=args.all_populations_dir,
            all_benchmarks_dir=args.all_benchmarks_dir,
            generation_num=args.generation_num,
            num_inference_steps=args.num_inference_steps,
            min_diff_from_default=args.min_diff_from_default,
            population_size=args.population_size,
            maximize_macs=args.maximize_MACs,
        )

    return manager


def get_offline_eval_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    """
    Get the common offline evaluation keyword arguments.
    """
    return {
        "embedding_dir": args.embedding_dir,
        "num_images_per_prompt": args.num_images_per_prompt,
        "exactly_n_images": args.exactly_n_images,
        "regen_if_not_n_images": args.regen_if_not_n_images,
        "benchmark_prompts": args.benchmark_prompts,
        "file_mode": args.file_mode,
        "image_naming_mode": args.image_naming_mode,
    }
