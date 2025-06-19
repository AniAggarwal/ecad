"""
Single GPU implementation of NSGA-II training with offline evaluation.
This version runs directly on a single GPU without requiring SLURM.
"""

import argparse
import subprocess
from typing import Any
from pathlib import Path

from ecad.genetic.train_nsga2_base import (
    get_base_argparser,
    validate_base_args,
    init_gen_0,
    train_one_cycle,
    checkpoint_manager_and_algorithm,
    initialize_manager,
    get_offline_eval_kwargs,
)
from ecad.genetic.population_io_manager import PopulationIOManager
from ecad.image_generators.load_image_generator import (
    get_image_generator_type,
)
from pymoo.algorithms.moo.nsga2 import NSGA2


def get_offline_eval_commands_single_gpu_from_manager(
    manager: PopulationIOManager,
    image_generator: str,
    print_commands: bool = True,
    batch_size: int | None = None,
    embedding_dir: Path | None = None,
    num_images_per_prompt: int | None = None,
    benchmark_prompts: Path | None = None,
    file_mode: str | None = None,
    image_naming_mode: str | None = None,
    exactly_n_images: int = 1000,
    regen_if_not_n_images: int = 1000,
) -> tuple[str, str, str]:
    """
    Given a PopulationIOManager instance, generate the offline evaluation commands
    for single GPU execution (non-SLURM) for:
      - Generating images for a generation.
      - Scoring the generated images.
      - Computing metrics from the candidates.

    The function uses the manager's name, generation number, and directory structure.
    """
    # Get the directories from the manager
    schedule_dir = manager.get_pop_candidates_dir()
    image_dir = manager.get_benchmark_gen_dir()

    # Build command for generating images using generate_images.py
    batch_size_str = (
        f"--batch-size {batch_size} " if batch_size is not None else ""
    )
    embedding_dir_str = (
        f"--input-dir {embedding_dir} " if embedding_dir is not None else ""
    )
    num_images_per_prompt_str = (
        f"--num-images-per-prompt {num_images_per_prompt} "
        if num_images_per_prompt is not None
        else ""
    )
    regen_str = (
        f"--regen-if-not-n-images {regen_if_not_n_images} "
        if regen_if_not_n_images is not None
        else ""
    )

    gen_images_cmd = (
        f"python ecad/benchmark/generate_images.py "
        f"--image-generator {image_generator} "
        f"{embedding_dir_str}"
        f"--output-dir {image_dir} "
        f"--schedule-dir {schedule_dir} "
        f"{num_images_per_prompt_str}"
        f"{batch_size_str}"
        f"{regen_str}"
    ).strip()

    # Build command for scoring images using score_images.py
    file_mode_str = (
        f"--file-mode {file_mode} " if file_mode is not None else ""
    )
    image_naming_mode_str = (
        f"--image-naming-mode {image_naming_mode} "
        if image_naming_mode is not None
        else ""
    )
    benchmark_prompts_str = (
        f"--benchmark-prompts {benchmark_prompts} "
        if benchmark_prompts is not None
        else ""
    )
    exactly_n_str = (
        f"--exactly-n-images {exactly_n_images} "
        if exactly_n_images is not None
        else ""
    )

    score_images_cmd = (
        f"python ecad/benchmark/score_images.py "
        f"{benchmark_prompts_str}"
        f"--image-dir {image_dir} "
        f"{exactly_n_str}"
        f"--delete-after "
        f"{file_mode_str}"
        f"{image_naming_mode_str}"
    ).strip()

    # Build command for computing metrics using compute_macs.py
    compute_metrics_cmd = (
        f"python ecad/benchmark/compute_macs.py "
        f"--image-generator {image_generator} "
        f"--input-dir {schedule_dir}"
    )

    if print_commands:
        print("Single GPU execution commands:")
        print("\nCommand to generate images:")
        print(gen_images_cmd)
        print("\nCommand to score images:")
        print(score_images_cmd)
        print("\nCommand to compute metrics:")
        print(compute_metrics_cmd)

    return gen_images_cmd, score_images_cmd, compute_metrics_cmd


def run_single_gpu_command(
    command: str, print_eval_outputs: bool = False
) -> None:
    """
    Execute a command directly on the current machine (single GPU setup).

    Parameters:
        command (str): The command to execute.

    Raises:
        RuntimeError: If the command fails.
    """
    try:
        print(f"\nExecuting: {command}")
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True, check=True
        )
        print(f"Command completed successfully")
        if print_eval_outputs and result.stdout:
            print(f"STDOUT: {result.stdout.strip()}")
        if result.stderr:
            print(f"STDERR: {result.stderr.strip()}")
    except subprocess.CalledProcessError as err:
        print(f"Command failed with exit code {err.returncode}")
        print(f"STDERR: {err.stderr.strip()}")
        raise RuntimeError(
            f"Command execution failed: {err.stderr.strip()}"
        ) from err


def train_nsga2_single_gpu(
    manager: PopulationIOManager,
    algorithm: NSGA2,
    image_generator: str,
    num_cycles: int | None,
    print_not_submit: bool = True,
    batch_size: int | None = None,
    offline_eval_kwargs: dict[str, Any] | None = None,
):
    """
    Training loop for single GPU execution (non-SLURM).

    Parameters:
        manager: PopulationIOManager instance
        algorithm: NSGA2 algorithm instance
        image_generator: Name of the image generator to use
        num_cycles: Number of cycles to run (None for infinite)
        print_not_submit: If True, only print commands without executing them
        batch_size: Batch size for image generation
        offline_eval_kwargs: Additional kwargs for offline evaluation
    """
    if offline_eval_kwargs is None:
        offline_eval_kwargs = {}

    failed = False
    while num_cycles is None or num_cycles > 0:
        step_taken = train_one_cycle(manager, algorithm)
        checkpoint_manager_and_algorithm(manager, algorithm)

        if not print_not_submit and not step_taken:
            failed = True

        print_eval_outputs = offline_eval_kwargs.pop(
            "print_eval_outputs", False
        )


        gen_images_cmd, score_images_cmd, compute_metrics_cmd = (
            get_offline_eval_commands_single_gpu_from_manager(
                manager,
                image_generator,
                print_commands=print_not_submit,
                batch_size=batch_size,
                **offline_eval_kwargs,
            )
        )

        if not print_not_submit:
            # Execute commands sequentially for single GPU (blocking execution)
            try:
                print("\n" + "=" * 80)
                print("Starting image generation...")
                print("=" * 80)
                run_single_gpu_command(gen_images_cmd, print_eval_outputs)

                print("\n" + "=" * 80)
                print("Starting image scoring...")
                print("=" * 80)
                run_single_gpu_command(score_images_cmd, print_eval_outputs)

                print("\n" + "=" * 80)
                print("Starting metrics computation...")
                print("=" * 80)
                run_single_gpu_command(compute_metrics_cmd, print_eval_outputs)

                print("\n" + "=" * 80)
                print("All offline evaluation steps completed successfully!")
                print("=" * 80)

            except RuntimeError as e:
                print(f"Error during offline evaluation: {e}")
                raise

        if num_cycles is not None:
            num_cycles -= 1

        if failed:
            # try once more
            step_taken = train_one_cycle(manager, algorithm)
            checkpoint_manager_and_algorithm(manager, algorithm)

            if not print_not_submit and not step_taken:
                raise ValueError("Offline evaluation not found.")

            failed = False


def parse_args():
    """Parse command line arguments for single GPU execution."""
    parser = argparse.ArgumentParser(
        description=(
            "Run n cycles of NSGA-II training with offline evaluation on a single GPU. "
            "Either provide a path to load an existing population manager or "
            "specify all other arguments to start a new run."
        ),
        parents=[get_base_argparser()],
    )

    parser.add_argument(
        "--print-eval-outputs",
        action="store_true",
        help="Print outputs of each offline evaluation command.",
    )

    args = parser.parse_args()
    validate_base_args(args)
    return args


def main():
    """Main entry point for single GPU execution."""
    args = parse_args()

    image_generator_type = get_image_generator_type(args.image_generator)
    manager = initialize_manager(args, image_generator_type)

    try:
        print("Attempting to load from checkpoint...")
        algorithm = manager.load_algorithm()
    except FileNotFoundError as e:
        print(f"Checkpoint not found: {e}")
        algorithm = init_gen_0(manager, image_generator_type)

    offline_eval_kwargs = get_offline_eval_kwargs(args)
    offline_eval_kwargs["print_eval_outputs"] = args.print_eval_outputs

    num_cycles = None if args.num_cycles == "inf" else args.num_cycles

    train_nsga2_single_gpu(
        manager,
        algorithm,
        args.image_generator,
        num_cycles,
        args.print_not_submit,
        args.batch_size,
        offline_eval_kwargs=offline_eval_kwargs,
    )


if __name__ == "__main__":
    main()
