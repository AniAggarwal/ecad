import argparse
from datetime import datetime
from pathlib import Path

from ecad.image_generators.load_image_generator import (
    ImageGeneratorRegistry,
    get_image_generator_type,
)
from ecad.image_generators.pixart_image_generator import (
    PixArtImageGenerator,
)

DEFAULT_EMBEDDING_NAME = "embedding.pt"
DEFAULT_IMAGE_NAME = "output.png"
DEFAULT_OUTPUT_DIR = Path("results/inference/")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inference with prompts or embeddings. "
        "Supply a prompt, prompt file, or a path to directory of prompt embedding files, but not all. "
        "Note that PixArt inference at resolution other than 256x256 should be done with a custom schedule that specifies the height, width, and 1024x1024/MS transformer weights.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "image_generator",
        metavar="image-generator",
        type=str,
        help="The name of the image generator to use.",
        choices=list(ImageGeneratorRegistry.registry.keys()),
    )
    parser.add_argument(
        "-s",
        "--schedule",
        type=Path,
        required=False,
        help="Path to the schedule JSON file containing the caching schedule matching the specified image generator. "
        "For uncached/baseline, do not use this flag.",
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "-p",
        "--prompt",
        type=str,
        required=False,
        help="The prompt for which to generate the image.",
    )
    input_group.add_argument(
        "-f",
        "--prompt-file",
        type=Path,
        required=False,
        help="The prompt file for which to generate images.",
    )
    input_group.add_argument(
        "-i",
        "--input-embeddings",
        type=Path,
        required=False,
        help="A path to a directory of prompt embedding file (e.g., 000_embedding.pt, 001_embedding.pt, etc.) to use for image generation.",
    )

    parser.add_argument(
        "-e",
        "--embedding-name",
        type=Path,
        default=DEFAULT_EMBEDDING_NAME,
        help=f"If a prompt is provided, the name of the file its embedding is saved to; should end in .pt.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR
        / datetime.now().replace(microsecond=0).isoformat(),
        help=f"Directory where the output subdirectory embeddings and images will be created and outputs saved to.",
    )

    parser.add_argument(
        "-n",
        "--num-images-per-prompt",
        type=int,
        default=10,
        help="Number of images to generate for each prompt.",
    )
    parser.add_argument(
        "--start-seed",
        type=int,
        required=False,
        default=0,
        help="Random seed used to generate the first image per prompt. "
        "Each subsequent image for that same prompt will use a seed incremented by seed-step amount",
    )
    parser.add_argument(
        "--seed-step",
        type=int,
        required=False,
        default=1,
        help="Amount the random seed used to generate each subsequent image for a given prompt is increased by.",
    )

    parser.add_argument(
        "--batch-size-embedding",
        type=int,
        required=False,
        default=1,
        help="Batch size for encoding prompts into embeddings.",
    )
    parser.add_argument(
        "--batch-size-generate",
        type=int,
        required=False,
        default=1,
        help="Batch size for generating images from embeddings.",
    )

    parser.add_argument(
        "--height",
        type=int,
        required=False,
        help="Height of the generated image. If specified, this overrides the schedule's height.",
    )
    parser.add_argument(
        "--width",
        type=int,
        required=False,
        help="Width of the generated image. If specified, this overrides the schedule's width.",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        required=False,
        help="Guidance scale for the image generation. If specified, this overrides the schedule's guidance scale.",
    )
    args = parser.parse_args()

    if args.schedule is not None and not args.schedule.exists():
        raise ValueError(f"Schedule file {args.schedule} not found.")

    if (
        sum(
            x is not None
            for x in (args.prompt, args.prompt_file, args.input_embeddings)
        )
        != 1
    ):
        raise ValueError(
            "You must provide exactly one of: prompt, prompt file, or a path to input embedding(s)."
        )

    if not args.output_dir.exists():
        print(f"Output directory {args.output_dir} not found. Creating.")
        args.output_dir.mkdir(parents=True)

    prompt_dir = args.output_dir / "embeddings"
    images_dir = args.output_dir / "images"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    img_gen_type = get_image_generator_type(args.image_generator)
    img_gen_kwargs = {}

    if args.start_seed is not None:
        img_gen_kwargs["start_seed"] = args.start_seed
    if args.seed_step is not None:
        img_gen_kwargs["seed_step"] = args.seed_step
    image_generator = img_gen_type(**img_gen_kwargs)

    # if prompts are provided instead of embeddings
    if args.prompt is not None or args.prompt_file is not None:
        if args.prompt is not None:
            name_to_prompt = {args.embedding_name.name: args.prompt}
        elif args.prompt_file is not None and args.prompt_file.exists():
            with open(args.prompt_file, "r") as f:
                prompts = [line.strip() for line in f if line.strip()]

            base_name = Path(args.embedding_name)
            name_to_prompt = {
                base_name.with_name(f"{i:03d}_{base_name.name}").name: prompt
                for i, prompt in enumerate(prompts)
            }
        else:
            raise ValueError("No prompt or prompt file provided or found.")

        # dump embeds
        print(f"Generating embeddings for prompts to {prompt_dir}.")
        image_generator.encode_and_save_prompts(
            name_to_prompt,
            prompt_dir,
            free_after=True,
            batch_size=args.batch_size_embedding,
        )
        print("Prompt file(s) saved.")

    else:
        # input embeddings provided
        prompt_dir = args.input_embeddings

    print(f"Using input embeddings from {prompt_dir}.")

    if not prompt_dir.exists() or not any(prompt_dir.glob("**/*.pt")):
        raise ValueError(f"No prompt embeddings found in {prompt_dir}.")

    inference_kwargs = {}
    if args.height is not None:
        inference_kwargs["height"] = args.height
    if args.width is not None:
        inference_kwargs["width"] = args.width
    if args.guidance_scale is not None:
        if img_gen_type == PixArtImageGenerator:
            raise ValueError(
                "PixArtImageGenerator does not support guidance scale. CFG is always enabled."
            )
        inference_kwargs["guidance_scale"] = args.guidance_scale

    image_generator.generate_from_saved_prompts(
        prompt_dir,
        images_dir,
        batch_size=args.batch_size_generate,
        images_per_prompt=args.num_images_per_prompt,
        free_after=True,
        **inference_kwargs,
    )


if __name__ == "__main__":
    main()
