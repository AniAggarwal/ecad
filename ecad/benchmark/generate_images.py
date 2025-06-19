import argparse
from pathlib import Path

from ecad.image_generators.image_generator import (
    ImageGenerator,
)
from ecad.image_generators.load_image_generator import (
    ImageGeneratorRegistry,
    get_image_generator_type,
)


def generate_for_schedule(
    image_generator_type: type[ImageGenerator],
    schedule_file: Path,
    embedding_dir: Path,
    output_dir: Path,
    batch_size: int,
    num_images_per_prompt: int,
    seed: int,
    seed_step: int,
    regen_not_n_imgs: int | None = None,
    dont_include_seed: bool = False,
):
    # skip if output dir exists and has the right number of images or
    # no number of images is specified and the output dir exists with at least 1 image
    if output_dir.exists() and (
        (found_imgs := len(list(output_dir.glob("*.png")))) > 0
    ):
        if regen_not_n_imgs is not None and found_imgs != regen_not_n_imgs:
            print(
                f"Regenerating images for schedule {schedule_file.stem}; expected {regen_not_n_imgs} images, found {found_imgs}."
            )
            for img in output_dir.glob("*.png"):
                img.unlink()

        # must have found exactly n images or no required number of images
        else:
            print(
                f"Skipping schedule {schedule_file.stem} because output directory already exists."
                f"Expected {regen_not_n_imgs if regen_not_n_imgs is not None else 'any number'}, found {found_imgs} images."
            )
            return

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n\n\nGenerating images for schedule: {schedule_file.stem}.\n")

    try:
        image_generator = image_generator_type(  # type: ignore
            start_seed=seed, seed_step=seed_step, schedule_path=schedule_file
        )
    except Exception as e:
        raise ValueError(
            "Error creating image generator, check arguments passed during init in the code."
        ) from e

    image_generator.generate_from_saved_prompts(
        embedding_dir,
        output_dir,
        batch_size=batch_size,
        images_per_prompt=num_images_per_prompt,
        include_seed_in_name=(not dont_include_seed),
    )


def generate_all_schedules(
    image_generator_type: type[ImageGenerator],
    schedule_dir: Path,
    embedding_dir: Path,
    output_dir: Path,
    batch_size: int,
    num_images_per_prompt: int,
    seed: int,
    seed_step: int,
    regen_not_n_imgs: int | None = None,
    dont_include_seed: bool = False,
):

    if schedule_dir.is_file() and schedule_dir.suffix == ".json":
        generate_for_schedule(
            image_generator_type,
            schedule_dir,
            embedding_dir,
            output_dir / schedule_dir.stem,
            batch_size,
            num_images_per_prompt,
            seed,
            seed_step,
            regen_not_n_imgs,
            dont_include_seed,
        )
        return

    # if schedule_dir only has dirs, then mirror its structure in the output_dir
    if all((subdir.is_dir() for subdir in schedule_dir.iterdir())):
        output_subdir = output_dir / schedule_dir.name
        for subdir in schedule_dir.iterdir():
            if subdir.is_dir():  # check again in case something changed
                generate_all_schedules(
                    image_generator_type,
                    subdir,
                    embedding_dir,
                    output_subdir,
                    batch_size,
                    num_images_per_prompt,
                    seed,
                    seed_step,
                    regen_not_n_imgs,
                    dont_include_seed,
                )
        return

    for schedule_file in schedule_dir.glob("**/*.json"):
        generate_for_schedule(
            image_generator_type,
            schedule_file,
            embedding_dir,
            output_dir / schedule_file.stem,
            batch_size,
            num_images_per_prompt,
            seed,
            seed_step,
            regen_not_n_imgs,
            dont_include_seed,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Use prompt embeddings to generate images."
        "A subdirectory under the output directory will be created for each JSON schedule found in the schedule directory."
    )
    parser.add_argument(
        "--image-generator",
        type=str,
        required=True,
        help="The name of the image generator to use.",
        choices=list(ImageGeneratorRegistry.registry.keys()),
    )
    parser.add_argument(
        "-i",
        "--input-dir",
        type=Path,
        help="Path to the directory containing prompt embeddings in <name>.pt format.",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        help="Directory to save generated images to."
        "A subdirectory will be created for each schedule found in the schedule directory.",
        required=True,
    )
    parser.add_argument(
        "-d",
        "--schedule-dir",
        type=Path,
        help="Directory to read schedules from in <schedule-name>.json format.",
        required=True,
    )
    parser.add_argument(
        "-n",
        "--num-images-per-prompt",
        type=int,
        default=10,
        help=f"The number of images to generate per prompt.",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=100,
        help=f"Batch size while generating images.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="The random seed to start each image generation with.",
    )
    parser.add_argument(
        "--seed-step",
        type=int,
        default=1,
        help="The amount to increment the seed by for each subsequent generation of the same prompt.",
    )
    parser.add_argument(
        "--regen-if-not-n-images",
        type=int,
        required=False,
        help="Force re-generating images for schedules whose directories do not contain exactly n images.",
    )
    parser.add_argument(
        "--dont-include-seed",
        action="store_true",
        help="Do not include the seed generated images names'.",
    )
    args = parser.parse_args()

    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input directory {args.input_dir} not found.")

    if not args.schedule_dir.exists():
        raise FileNotFoundError(
            f"Directory {args.schedule_dir} for schedules not found."
        )

    if not args.output_dir.exists():
        print(f"Output directory {args.output_dir} not found. Creating.")
        args.output_dir.mkdir(parents=True, exist_ok=True)

    image_generator_type = get_image_generator_type(args.image_generator)

    generate_all_schedules(
        image_generator_type,
        args.schedule_dir,
        args.input_dir,
        args.output_dir,
        args.batch_size,
        args.num_images_per_prompt,
        args.seed,
        args.seed_step,
        args.regen_if_not_n_images,
        args.dont_include_seed,
    )

    print("Done.")


if __name__ == "__main__":
    main()
