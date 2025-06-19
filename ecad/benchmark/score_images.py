import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import ImageReward as RM
import torch
import pandas as pd

from ecad.benchmark.generate_embeddings import read_benchmark_prompts

DEFAULT_BENCHMARK_DIR = Path(__file__).parent / "../../results/benchmark/"
DEFAULT_BENCHMARK_PROMPTS = DEFAULT_BENCHMARK_DIR / "benchmark-prompts.json"
DEFAULT_IMAGE_DIR = DEFAULT_BENCHMARK_DIR / "full-benchmark" / "images"

# Pattern to extract prompt id and image seed from filenames.
FILENAME_PATTERN = re.compile(
    r".*__prompt_id:(?P<prompt_id>.+?)__.*?__image_seed:(?P<image_seed>\d+)"
)
FILENAME_PATTERN_PARTI = re.compile(
    r"(?P<prompt_num>\d+)__prompt_seed:(?P<prompt_seed>.+?)__image_seed:(?P<image_seed>\d+)"
)
FILENAME_PATTERN_TOCA = re.compile(r"(?P<prompt_num>\d+)__.*")
FILENAME_PATTERN_TOCA_SEEDED = re.compile(
    r"(?P<prompt_num>\d+)__.*?image_seed:(?P<image_seed>\d+)"
)


def load_image_reward_model() -> RM.ImageReward:
    print("Loading ImageReward model.")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model: RM.ImageReward = RM.load("ImageReward-v1.0", device=device)
    model = torch.compile(model, fullgraph=True)  # type: ignore

    print("Model loaded.")
    return model


def save_to_file(input_path: Path, output_subpath: Path, data: dict[str, Any]):
    output_path = input_path / output_subpath
    # Ensure parent directories exist.
    output_path.parent.mkdir(parents=True, exist_ok=True)

    i = 1
    while output_path.exists():
        print(f"File {output_path} already exists.")
        output_path = output_path.parent / (
            output_path.stem + f"({i})" + output_path.suffix
        )
        i += 1

    print(f"Saving score data to {output_path}.")
    with open(output_path, "w") as f:
        json.dump(data, f, indent=4)


def fname_to_info(input_dir: Path) -> dict[str, dict[str, str]]:
    print(f"Reading from {input_dir}.\nUsing pattern {FILENAME_PATTERN}.\n")
    fname_to_prompt_id = {}
    for file in input_dir.glob("*.png"):
        name = file.stem
        match = FILENAME_PATTERN.match(name)
        if match is None:
            print(f"Invalid filename: {name}")
            continue
        prompt_id = match.group("prompt_id")
        image_seed = match.group("image_seed")
        fname_to_prompt_id[str(file.resolve())] = {
            "prompt_id": prompt_id,
            "image_seed": image_seed,
        }
    return fname_to_prompt_id


@torch.inference_mode()
def score_dir(
    input_dir: Path,
    prompt_id_to_text: dict[str, str] | list[str],
    model: RM.ImageReward,
    image_naming_mode: str,
) -> dict[str, dict[int, float]]:
    if image_naming_mode == "image_reward":
        return score_dir_image_reward(input_dir, prompt_id_to_text, model)
    elif image_naming_mode == "toca":
        return score_dir_toca(input_dir, prompt_id_to_text, model)
    elif image_naming_mode == "parti":
        return score_dir_parti(input_dir, prompt_id_to_text, model)

    raise ValueError(f"Got invalid image naming mode: {image_naming_mode}")


@torch.inference_mode()
def score_dir_image_reward(
    input_dir: Path,
    prompt_id_to_text: dict[str, str],
    model: RM.ImageReward,
) -> dict[str, dict[int, float]]:
    infos = fname_to_info(input_dir)
    score_by_prompt_id = defaultdict(dict)
    for fname, info in infos.items():
        prompt_id = info["prompt_id"]  # type: ignore
        image_seed = info["image_seed"]  # type: ignore
        prompt = prompt_id_to_text[prompt_id]
        score = model.score(prompt, fname)
        score_by_prompt_id[prompt_id][image_seed] = score
    return score_by_prompt_id


@torch.inference_mode()
def score_dir_parti(
    input_dir: Path,
    prompt_num_to_text: list[str],
    model: RM.ImageReward,
) -> dict[str, dict[int, float]]:
    print("Scoring Parti prompts now...")
    score_by_prompt_id = defaultdict(dict)

    for file in input_dir.glob("*.png"):
        name = file.stem
        match = FILENAME_PATTERN_PARTI.match(name)
        if match is None:
            print(f"Invalid filename: {name}")
            continue
        prompt_id = int(match.group("prompt_num"))
        image_seed = int(match.group("image_seed"))
        prompt = prompt_num_to_text[prompt_id]
        score = model.score(prompt, str(file.resolve()))
        score_by_prompt_id[prompt_id][image_seed] = score

    return score_by_prompt_id


@torch.inference_mode()
def score_dir_toca(
    input_dir: Path,
    prompt_num_to_text: list[str],
    model: RM.ImageReward,
) -> dict[str, dict[int, float]]:
    print("Scoring ToCa prompts now...")
    score_by_prompt_id = defaultdict(dict)

    for file in input_dir.glob("*.png"):
        name = file.stem
        match = FILENAME_PATTERN_TOCA_SEEDED.match(name)
        if match is None:
            match = FILENAME_PATTERN_TOCA.match(name)
            if match is None:
                print(f"Invalid filename: {name}")
                continue
            seed = 0
        else:
            seed = int(match.group("image_seed"))

        prompt_id = int(match.group("prompt_num"))
        prompt = prompt_num_to_text[prompt_id]
        score = model.score(prompt, str(file.resolve()))
        score_by_prompt_id[prompt_id][seed] = score

    return score_by_prompt_id


def avg_per_prompt(
    score_by_prompt_id: dict[str, dict[int, float]]
) -> dict[str, float]:
    return {
        prompt_id: sum(info.values()) / len(info)
        for prompt_id, info in score_by_prompt_id.items()
    }


def avg_overall(score_by_prompt_id: dict[str, dict[int, float]]) -> float:
    total = sum(sum(info.values()) for info in score_by_prompt_id.values())
    num = sum(len(info) for info in score_by_prompt_id.values())
    print(
        f"Summed score across all images: {total}, number of images scored: {num}"
    )
    if num == 0:
        print("ERROR: No scores found.")
        return 0
    return total / num


def score_dirs_recursive(
    input_dir: Path,
    output_subpath: Path,
    prompt_id_to_text: dict[str, str],
    model: RM.ImageReward,
    image_naming_mode: str,
    delete_after: bool = False,
    exactly_n_images: int | None = None,
    rescore_existing: bool = False,
) -> None:
    if not input_dir.is_dir():
        return

    num_imgs = len(list(input_dir.glob("*.png")))
    if num_imgs > 0:
        if exactly_n_images is not None and num_imgs != exactly_n_images:
            print(
                f"ERROR: Directory {input_dir} has wrong number of images. Expected {exactly_n_images}, found {num_imgs}."
            )
        elif not rescore_existing and (input_dir / output_subpath).exists():
            print(f"Skipping {input_dir} because it already has a score file.")
        else:
            score_by_prompt_id = score_dir(
                input_dir,
                prompt_id_to_text,
                model,
                image_naming_mode,
            )
            avg_by_prompt = avg_per_prompt(score_by_prompt_id)
            total_score = avg_overall(score_by_prompt_id)
            full_data = {
                "total_score": total_score,
                "avg_by_prompt": avg_by_prompt,
                "score_by_prompt_id": score_by_prompt_id,
            }
            save_to_file(input_dir, output_subpath, full_data)
            print(f"Total Score Data for {input_dir}: {total_score}")
            if delete_after:
                print(f"Deleting images in {input_dir}.")
                for file in input_dir.glob("*.png"):
                    file.unlink()

    for sub_dir in input_dir.iterdir():
        score_dirs_recursive(
            sub_dir,
            output_subpath,
            prompt_id_to_text,
            model,
            image_naming_mode,
            delete_after,
            exactly_n_images,
        )


def get_score_images_argparser() -> argparse.ArgumentParser:
    """
    Returns an ArgumentParser preloaded with score-images–specific arguments.
    This parser can be used standalone or merged into a multi-job parser.
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--benchmark-prompts",
        type=Path,
        default=DEFAULT_BENCHMARK_PROMPTS,
        help="JSON file containing benchmark prompts.",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        help="Directory containing images to be scored.",
    )
    parser.add_argument(
        "--output-subpath",
        "-o",
        type=Path,
        default=Path("scores.json"),
        help="Filename or subpath for the generated score JSON file.",
    )
    parser.add_argument(
        "--delete-after",
        action="store_true",
        help="Delete the images within the directory after scoring.",
    )
    parser.add_argument(
        "--exactly-n-images",
        "-n",
        type=int,
        help="Only score directories with exactly n images.",
    )
    parser.add_argument(
        "--rescore-existing",
        action="store_true",
        help="Rescore directories that already have a score file.",
    )
    parser.add_argument(
        "--image-naming-mode",
        choices=["image_reward", "parti", "toca"],
        default="image_reward",
        help="The pattern to use to match image filenames.",
    )
    parser.add_argument(
        "--file-mode",
        choices=["json", "text", "txt", "tsv"],
        help="JSON is the Image Reward prompts mode. TSV must have a column named 'Prompt', and have a header line. Text format is a simple text file with a prompt on each line.",
    )
    return parser


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score benchmark images using the ImageReward model."
    )
    score_parser = get_score_images_argparser()
    # Merge score-specific arguments into the main parser.
    for action in score_parser._actions:
        parser._add_action(action)
    args = parser.parse_args()

    if not args.benchmark_prompts.exists():
        raise FileNotFoundError(f"File {args.benchmark_prompts} not found.")
    
    if args.file_mode is None:
        args.file_mode = args.benchmark_prompts.suffix[1:].split(".")[-1]

    if args.file_mode.lower() == "json":
        prompt_json = read_benchmark_prompts(args.benchmark_prompts)
        if prompt_json is None:
            raise ValueError("ERROR: no prompt data found.")
        prompt_id_to_text = {
            item["id"]: item["prompt"] for item in prompt_json
        }
    elif args.file_mode.lower() == "tsv":
        df = pd.read_csv(args.benchmark_prompts, sep="\t")
        prompt_id_to_text = {
            i: prompt for i, prompt in enumerate(df["Prompt"].tolist())
        }
    elif args.file_mode.lower() in ("text", "txt"):
        with open(args.benchmark_prompts, "r") as f:
            prompt_id_to_text = {
                i: line.strip() for i, line in enumerate(f.readlines())
            }
    else:
        raise ValueError(
            f"Unsupported file type: {args.benchmark_prompts.suffix}. "
        )

    if args.image_naming_mode == "toca" and args.file_mode.lower() == "json":
        raise ValueError(
            "ToCa mode requires a text or TSV file. JSON is not supported yet."
        )

    model = load_image_reward_model()
    print("Model loaded; beginning scoring...")
    score_dirs_recursive(
        args.image_dir,
        args.output_subpath,
        prompt_id_to_text,
        model,
        args.image_naming_mode,
        args.delete_after,
        args.exactly_n_images,
        args.rescore_existing,
    )
    print("Done.")


if __name__ == "__main__":
    main()
