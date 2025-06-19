import argparse
import json
from collections import defaultdict
from pathlib import Path

from diffusers.utils import logging

from ecad.benchmark.generate_embeddings import batch_dict
from ecad.image_generators.image_generator import ImageGenerator
from ecad.image_generators.load_image_generator import (
    ImageGeneratorRegistry,
    get_image_generator_type,
)


def read_mjhq_prompts(
    file_path: Path,
) -> dict[str, dict[str, str]] | None:
    print(f"Reading prompts from {file_path}.")

    data = None
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"File {file_path} not found.")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")

    if data is None:
        return None

    cat_to_name_to_prompt = defaultdict(dict)
    for name, inner_dict in data.items():
        cat_to_name_to_prompt[inner_dict["category"]][name] = inner_dict[
            "prompt"
        ]

    return cat_to_name_to_prompt


def generate_mjhq_embeddings(
    image_generator_type: type[ImageGenerator],
    meta_prompts_file: Path,
    output_dir: Path,
    batch_size: int | None,
    seed: int,
    seed_step: int,
) -> None:
    prompts_data = read_mjhq_prompts(meta_prompts_file)
    if prompts_data is None:
        print("No prompts data found.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        # note we don't pass a schedule file since we are only generating embeddings
        image_generator = image_generator_type(  # type: ignore
            start_seed=seed, seed_step=seed_step
        )
    except Exception as e:
        raise ValueError(
            "Error creating image generator, check arguments passed during init in the code."
        ) from e

    # since MJHQ has large prompts that are truncated, which cause LONG warnings
    prior_verbosity = logging.get_verbosity()
    logging.set_verbosity_error()

    for category, name_to_prompt in prompts_data.items():
        print(f"Generating {len(name_to_prompt)} for category: {category}")
        cat_dir = output_dir / category
        cat_dir.mkdir(parents=True, exist_ok=True)

        # batch name_to_prompt to avoid OOM
        if batch_size is not None:
            batches = batch_dict(name_to_prompt, batch_size)
        else:
            batches = [name_to_prompt]
        for i, batch in enumerate(batches):
            print(
                f"Generating embeddings for batch {i + 1} of {len(batches)}."
            )
            image_generator.encode_and_save_prompts(
                batch, cat_dir, batch_size=batch_size
            )

    logging.set_verbosity(prior_verbosity)
    print("Done.")


def main():
    parser = argparse.ArgumentParser(
        description="Generate embeddings for MJHQ-30K prompts to compute FID."
        "A subdirectory under the output directory will be created for each MJHQ image category."
    )
    parser.add_argument(
        "--image-generator",
        type=str,
        required=True,
        help="The name of the image generator to use.",
        choices=list(ImageGeneratorRegistry.registry.keys()),
    )
    parser.add_argument(
        "-f",
        "--file-path",
        type=Path,
        required=True,
        help="Path to the prompts JSON file.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to save generated prompt embeddings to.",
    )
    parser.add_argument(
        "--prompt-batch-size",
        type=int,
        default=1000,
        help="Number of prompts to generate embeddings for in each batch.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="The random seed to generate with.",
    )
    parser.add_argument(
        "--seed-step",
        type=int,
        default=0,
        help="The amount to step the seed by for each prompt.",
    )
    args = parser.parse_args()

    if not args.file_path.exists():
        print(f"Benchmark file {args.file_path} not found.")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)

    image_generator_type = get_image_generator_type(args.image_generator)
    generate_mjhq_embeddings(
        image_generator_type,
        args.file_path,
        args.output_dir,
        args.prompt_batch_size,
        args.seed,
        args.seed_step,
    )


if __name__ == "__main__":
    main()
