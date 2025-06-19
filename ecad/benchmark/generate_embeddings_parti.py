import argparse
import pandas as pd
from pathlib import Path

from ecad.image_generators.load_image_generator import (
    ImageGeneratorRegistry,
    get_image_generator_type,
)
from ecad.image_generators.pixart_image_generator import (
    ImageGenerator,
)


def read_benchmark_prompts(file_path: Path) -> list[str] | None:
    print(f"Reading prompts from {file_path}.")

    data = None
    try:
        df = pd.read_csv(file_path, sep="\t")
        data = df["Prompt"].to_list()
    except FileNotFoundError:
        print(f"File {file_path} not found.")
    except Exception as e:
        print(f"Error decoding: {e}")
    finally:
        return data


def generate_benchmark_embeddings(
    image_generator_type: type[ImageGenerator],
    benchmark_file: Path,
    output_dir: Path,
    seed: int,
    batch_size: int | None = None,
) -> None:
    prompts_data = read_benchmark_prompts(benchmark_file)
    if prompts_data is None:
        print("No prompts data found.")
        return

    # create a map of image names to their prompts
    name_to_prompt = {
        f"{i:04}__prompt_seed:{seed:03}": item
        for i, item in enumerate(prompts_data)
    }

    image_generator = image_generator_type(start_seed=seed, seed_step=0)
    image_generator.encode_and_save_prompts(
        name_to_prompt, output_dir, batch_size=batch_size
    )


def main():
    parser = argparse.ArgumentParser(
        description="Process PartiPrompts and generate embeddings."
    )
    parser.add_argument(
        "--image-generator",
        type=str,
        required=False,
        help="The name of the image generator to use.",
        choices=list(ImageGeneratorRegistry.registry.keys()),
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=0,
        help="The random seed to generate with.",
    )
    parser.add_argument(
        "-f",
        "--file-path",
        type=Path,
        help="Path to the benchmark prompts tsv file.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        help="Directory to save generated prompt embeddings to.",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        help="Batch size for generating embeddings. If not provided, all prompts will be processed in one batch.",
    )

    args = parser.parse_args()

    if not args.file_path.exists():
        print(f"Benchmark file {args.file_path} not found.")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)

    image_generator_type = get_image_generator_type(args.image_generator)
    generate_benchmark_embeddings(
        image_generator_type,
        args.file_path,
        args.output_dir,
        args.seed,
        args.batch_size,
    )


if __name__ == "__main__":
    main()
