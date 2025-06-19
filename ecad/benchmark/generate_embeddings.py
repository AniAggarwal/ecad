import argparse
from itertools import islice
import json
from pathlib import Path

from ecad.image_generators.load_image_generator import (
    ImageGeneratorRegistry,
    get_image_generator_type,
)
from ecad.image_generators.pixart_image_generator import (
    ImageGenerator,
)


def read_benchmark_prompts(file_path: Path) -> list[dict[str, str]] | None:
    print(f"Reading prompts from {file_path}.")

    data = None
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"File {file_path} not found.")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
    finally:
        return data


def read_prompts_txt(file_path: Path) -> list[str] | None:
    print(f"Reading prompts from {file_path}.")
    data = None
    try:
        with open(file_path, "r") as f:
            data = f.readlines()
    except FileNotFoundError:
        print(f"File {file_path} not found.")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
    finally:
        return data


def generate_benchmark_embeddings(
    image_generator_type: type[ImageGenerator],
    benchmark_file: Path,
    output_dir: Path,
    seed: int,
    batch_size: int | None,
) -> None:
    if benchmark_file.suffix == ".txt":
        prompts_data = read_prompts_txt(benchmark_file)
        name_to_prompt = {
            f"{i:03d}__prompt_seed:{seed:03}": prompt
            for i, prompt in enumerate(prompts_data)
        }
    elif benchmark_file.suffix == ".json":
        prompts_data = read_benchmark_prompts(benchmark_file)
        # create a map of image names to their prompts
        name_to_prompt = {
            f"{i:03}__prompt_id:{item['id']}__prompt_seed:{seed:03}": item[
                "prompt"
            ]
            for i, item in enumerate(prompts_data)
        }
    else:
        raise ValueError(
            f"Unsupported file type: {benchmark_file.suffix}. Only .json and .txt are supported."
        )

    if prompts_data is None:
        print("No prompts data found.")
        return

    image_generator = image_generator_type(start_seed=seed, seed_step=0)

    # batch name_to_prompt to avoid OOM
    if batch_size is not None:
        batches = batch_dict(name_to_prompt, batch_size)
    else:
        batches = [name_to_prompt]

    for i, batch in enumerate(batches):
        print(f"Generating embeddings for batch {i + 1} of {len(batches)}.")
        image_generator.encode_and_save_prompts(
            batch, output_dir, batch_size=batch_size
        )


def batch_dict(
    orig_dict: dict[str, str], batch_size: int
) -> list[dict[str, str]]:
    """
    Split `orig_dict` into a list of dictionaries, each with up to `batch_size`
    key–value pairs, preserving the original order.
    """
    it = iter(orig_dict.items())
    batches: list[dict[str, str]] = []

    while True:
        # Take the next `batch_size` items
        chunk = dict(islice(it, batch_size))
        if not chunk:
            break
        batches.append(chunk)

    return batches


def main():
    parser = argparse.ArgumentParser(
        description="Process benchmark prompts and generate embeddings."
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
        help="Path to the benchmark prompts JSON file.",
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
        help="The batch size to use for generating embeddings. Do not use for batch size of all at once.",
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
