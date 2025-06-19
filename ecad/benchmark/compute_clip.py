import argparse
import json
import re
from pathlib import Path
import subprocess
from typing import Any
import sys

import ImageReward as RM
import pandas as pd
from tqdm import tqdm

from ecad.benchmark.generate_embeddings import read_benchmark_prompts
from ecad.benchmark.score_images import save_to_file

IMAGE_EXTENSIONS = ["png", "jpg", "jpeg"]

FILENAME_PATTERN_IM = re.compile(
    r"(?P<prompt_num>\d+)__prompt_id:(?P<prompt_id>.+?)__.*?__image_seed:(?P<image_seed>\d+)"
)
FILENAME_PATTERN_PARTI = re.compile(
    r"(?P<prompt_num>\d+)__prompt_seed:(?P<prompt_seed>.+?)__image_seed:(?P<image_seed>\d+)"
)
FILENAME_PATTERN_TOCA = re.compile(r"(?P<prompt_num>\d+)__.*")
FILENAME_PATTERN_MJHQ = re.compile(r"(?P<prompt_num>.*)")
FILENAME_PATTERN_COCO = re.compile(r"(?P<prompt_num>\d+)")
FILENAME_PATTERNS = {
    "image_reward": FILENAME_PATTERN_IM,
    "parti": FILENAME_PATTERN_PARTI,
    "toca": FILENAME_PATTERN_TOCA,
    "mjhq": FILENAME_PATTERN_MJHQ,
    "coco": FILENAME_PATTERN_COCO,
}

DEFAULT_PYTHON_PATH = Path(sys.executable).resolve()


def load_prompts(prompts_path: Path, file_mode: str) -> dict[int, str]:
    prompt_num_to_text = {}

    if file_mode.lower() == "json":
        prompt_json = read_benchmark_prompts(prompts_path)
        if prompt_json is None:
            raise ValueError(
                f"Failed to read prompts from {prompts_path}. "
                "Ensure the file is in the correct format."
            )
        prompt_num_to_text = {
            i: item["prompt"] for i, item in enumerate(prompt_json)
        }

    elif file_mode.lower() == "tsv":
        prompt_df = pd.read_csv(prompts_path, sep="\t")
        prompt_num_to_text = {
            i: prompt for i, prompt in enumerate(prompt_df["Prompt"].tolist())
        }
    elif file_mode.lower() == "text":
        with open(prompts_path, "r") as f:
            prompt_num_to_text = {
                i: line.strip() for i, line in enumerate(f.readlines())
            }
    else:
        raise ValueError("Invalid file mode. Choose 'json', 'tsv', or 'text'.")

    return prompt_num_to_text


def load_prompts_mjhq(prompts_path: Path) -> dict[str, str]:

    with open(prompts_path, "r") as f:
        prompt_json = json.load(f)

    prompt_id_to_text = {}

    for id, info in prompt_json.items():
        prompt_id_to_text[id] = info["prompt"]
    return prompt_id_to_text


def load_prompts_mjhq_as_nums(prompts_path: Path) -> dict[int, str]:
    with open(prompts_path, "r") as f:
        prompt_json = json.load(f)

    prompt_num_to_text = {}
    for i, info in enumerate(prompt_json.values()):
        prompt_num_to_text[i] = info["prompt"]

    return prompt_num_to_text


def match_imgs_to_prompts(
    image_dir: Path,
    prompt_num_to_text: dict[int, str] | dict[str, str],
    image_naming_mode: str,
) -> list[tuple[Path, str]]:
    imgs_to_prompts = []

    img_paths = []
    for ext in IMAGE_EXTENSIONS:
        img_paths.extend(image_dir.glob(f"**/*.{ext}"))

    img_paths.sort(key=lambda x: x.stem)

    pat = FILENAME_PATTERNS[image_naming_mode]

    for img_path in img_paths:
        prompt_num = None
        name = img_path.stem
        match = pat.match(name)

        if match is not None:
            prompt_num = match.group("prompt_num")

            if image_naming_mode != "mjhq":
                prompt_num = int(prompt_num)

        else:
            print(f"Invalid filename: {img_path.name}")
            continue

        if prompt_num not in prompt_num_to_text:
            print(
                f"Prompt number {prompt_num} not found in prompts. Filename: {img_path}."
            )
            continue

        imgs_to_prompts.append((img_path, prompt_num_to_text[prompt_num]))

    return imgs_to_prompts


def del_dir(dir_path: Path):
    if dir_path.exists():
        for item in dir_path.iterdir():
            if item.is_dir():
                del_dir(item)
            else:
                item.unlink()
        dir_path.rmdir()


def create_tmp_dir(
    image_dir: Path,
    img_paths_to_prompts: list[tuple[Path, str]],
) -> Path:
    tmp_dir = image_dir / "tmp_clip_dir/"
    print(f"Creating temporary directory: {tmp_dir}")

    if tmp_dir.exists():
        print(f"Temporary directory {tmp_dir} already exists. Removing it.")
        del_dir(tmp_dir)
    tmp_dir.mkdir()

    tmp_imgs = tmp_dir / "images"
    tmp_prompts = tmp_dir / "prompts"

    tmp_imgs.mkdir()
    tmp_prompts.mkdir()

    for i, (img_path, prompt) in tqdm(
        enumerate(img_paths_to_prompts), total=len(img_paths_to_prompts)
    ):
        suffix = img_path.suffix

        sym_path = tmp_imgs / f"{i:04}{suffix}"
        ext = 1
        while sym_path.exists():
            sym_path = tmp_imgs / f"{i:04}_{ext}{suffix}"
            ext += 1

        sym_path.symlink_to(img_path, False)

        prompt_path = tmp_prompts / sym_path.with_suffix(".txt").name
        with open(prompt_path, "w") as f:
            f.write(prompt)

    return tmp_dir


def compute_clip(tmp_dir: Path, python_path: Path, clip_model: str) -> float:
    print("Computing CLIP score...")
    command = [
        str(python_path),
        "-m",
        "clip_score",
        str(tmp_dir / "images"),
        str(tmp_dir / "prompts"),
        "--clip-model",
        clip_model,
        "--batch-size",
        "1",
    ]

    run_res = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        text=True,
    )
    print(f"STDOUT: {run_res.stdout.strip()}")

    if run_res.returncode != 0:
        raise ValueError(
            f"ERROR: failed to compute CLIP: {run_res.returncode}"
        )

    match = re.search(r"CLIP Score: (?P<score>.+)$", run_res.stdout)
    if match is None:
        raise ValueError(
            f"ERROR: failed to parse CLIP score: {run_res.stdout}"
        )

    score = float(match.group("score"))
    print(f"CLIP Score: {score}")
    return score


def score_dir(
    img_dir: Path,
    prompts_path: Path,
    tmp_dir: Path,
    python_path: Path,
    clip_model: str,
) -> dict[str, Any]:
    score = compute_clip(tmp_dir, python_path, clip_model)

    data = {
        "clip_score": score,
        "clip_model": clip_model,
        "img_dir": str(img_dir),
        "prompts": str(prompts_path),
    }
    return data


def get_args() -> argparse.ArgumentParser:
    """
    Returns an ArgumentParser preloaded with clip–specific arguments.
    This parser can be used standalone or merged into a multi-job parser.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--python-path",
        type=Path,
        default=DEFAULT_PYTHON_PATH,
        help="Path to a Python executable who's environment contains clip_score.",
    )
    parser.add_argument(
        "--prompts-path",
        type=Path,
        required=True,
        help="The file containing prompts.",
    )
    parser.add_argument(
        "-i",
        "--image-dir",
        type=Path,
        required=True,
        help="Directory containing images to be scored.",
    )
    parser.add_argument(
        "-m",
        "--file-mode",
        choices=["json", "text", "tsv"],
        required=True,
        help="JSON is the Image Reward prompts mode. TSV must have a column named 'Prompt', and have a header line. Text format is a simple text file with a prompt on each line.",
    )
    parser.add_argument(
        "-o",
        "--output-subpath",
        type=Path,
        default=Path("clip_scores.json"),
        help="Filename or subpath for the generated score JSON file.",
    )
    parser.add_argument(
        "--clip-model",
        type=str,
        default="openai/clip-vit-base-patch32",
        help="The CLIP model to use.",
    )
    parser.add_argument(
        "--mjhq",
        action="store_true",
        help="Use MJHQ directory format instead of standard, and MJHQ prompt file format.",
    )
    parser.add_argument(
        "--image-naming-mode",
        choices=FILENAME_PATTERNS,
        default="image_reward",
        help="The pattern to use to match image filenames.",
    )
    return parser


def main(args):
    prompt_to_text: dict[int, str] | dict[str, str]

    if args.mjhq:
        if args.image_naming_mode == "mjhq":
            prompt_to_text = load_prompts_mjhq(args.prompts_path)
        else:
            prompt_to_text = load_prompts_mjhq_as_nums(args.prompts_path)
    else:
        prompt_to_text = load_prompts(args.prompts_path, args.file_mode)

    imgs_to_prompts = match_imgs_to_prompts(
        args.image_dir, prompt_to_text, args.image_naming_mode
    )
    tmp_dir = create_tmp_dir(args.image_dir, imgs_to_prompts)
    print(f"Created tmp dir: {tmp_dir}")

    data = score_dir(
        args.image_dir,
        args.prompts_path,
        tmp_dir,
        args.python_path,
        args.clip_model,
    )

    save_to_file(args.image_dir, args.output_subpath, data)

    print(f"Deleting tmp dir: {tmp_dir}")
    del_dir(tmp_dir)


if __name__ == "__main__":
    parser = get_args()
    args = parser.parse_args()

    if not args.prompts_path.exists():
        raise FileNotFoundError(f"Prompts file {args.prompts_path} not found.")

    if not args.image_dir.exists() or not args.image_dir.is_dir():
        raise FileNotFoundError(f"Image directory {args.image_dir} not found.")

    main(args)
