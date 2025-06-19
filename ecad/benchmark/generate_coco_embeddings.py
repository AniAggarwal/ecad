import argparse
import json
from collections import defaultdict
from pathlib import Path

from diffusers.utils import logging
from ecad.image_generators.image_generator import ImageGenerator
from ecad.image_generators.load_image_generator import (
    ImageGeneratorRegistry,
    get_image_generator_type,
)


def read_coco_prompts(
    file_path: Path,
) -> dict[str, dict[str, str]] | None:
    print(f"Reading prompts from {file_path}.")

    stripped_lines = []
    with open(file_path, "r") as f:
        lines = f.readlines()
    for idx, line in enumerate(lines):
        stripped_lines.append(line.strip())

    prompts_data = {}
    for i in range(10):
        prompts_data[i] = {}
        for j in range(i * 3000, i * 3000 + 3000):
            prompts_data[i][i * 3000 + j] = stripped_lines[j]

    return prompts_data


def generate_coco_embeddings(
    image_generator_type: type[ImageGenerator],
    meta_prompts_file: Path,
    output_dir: Path,
    batch_size,
    seed: int,
    seed_step: int,
) -> None:
    prompts_data = read_coco_prompts(meta_prompts_file)
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

    prior_verbosity = logging.get_verbosity()
    logging.set_verbosity_error()

    for i in range(10):
        print(f"Encoding from {i*3000} - {i*3000 + 2999}")
        category = f"megabatch_{i}"
        cat_dir = output_dir / category
        cat_dir.mkdir(parents=True, exist_ok=True)

        image_generator.encode_and_save_prompts(
            prompts_data[i], cat_dir, batch_size=batch_size
        )

    logging.set_verbosity(prior_verbosity)
    print("Done.")


def main():
    parser = argparse.ArgumentParser(
        description="Generate embeddings for COCOref-30K prompts to compute FID."
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
        help="Path to the prompts txt file.",
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
    generate_coco_embeddings(
        image_generator_type,
        args.file_path,
        args.output_dir,
        args.prompt_batch_size,
        args.seed,
        args.seed_step,
    )


if __name__ == "__main__":
    main()
