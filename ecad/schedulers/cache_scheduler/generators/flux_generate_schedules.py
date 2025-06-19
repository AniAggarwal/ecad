import argparse
from pathlib import Path

from ecad.schedulers.cache_scheduler.generators.flux_schedule_generators import (
    GEN_FUNCTIONS,
)
from ecad.schedulers.cache_scheduler.generators.helpers import (
    save_schedules,
)

DEFAULT_SCHEDULE_DIR = Path("schedules/flux_cache_schedules/")
DEFAULT_FUNCTIONS = ["gen_default"]
DEFAULT_NUM_BLOCKS = 19
DEFAULT_NUM_SINGLE_BLOCKS = 38
DEFAULT_NUM_INFERENCE_STEPS = 20


def main():
    parser = argparse.ArgumentParser(
        description="Generate and save caching schedules."
    )
    parser.add_argument(
        "-p",
        "--path",
        type=Path,
        default=DEFAULT_SCHEDULE_DIR,
        help="Path to save the generated schedules. A directory will be created for the schedules.",
    )
    parser.add_argument(
        "--num_blocks",
        type=int,
        default=DEFAULT_NUM_BLOCKS,
        help="Number of blocks to use in the schedule generation.",
    )
    parser.add_argument(
        "--num_single_blocks",
        type=int,
        default=DEFAULT_NUM_SINGLE_BLOCKS,
        help="Number of single blocks to use in the schedule generation.",
    )
    parser.add_argument(
        "-s",
        "--num_inference_steps",
        type=int,
        default=DEFAULT_NUM_INFERENCE_STEPS,
        help="Number of inference steps to use in the schedule generation.",
    )
    parser.add_argument(
        "--height",
        type=int,
        required=False,
        help="Height of the image to be generated. If not provided, an internal default value is used.",
    )
    parser.add_argument(
        "--width",
        type=int,
        required=False,
        help="Width of the image to be generated. If not provided, an internal default value is used.",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        required=False,
        help="The guidance scale to use. If not provided, an internal default value is used.",
    )
    parser.add_argument(
        "-f",
        "--functions",
        nargs="+",
        default=DEFAULT_FUNCTIONS,
        help=f"List of functions to run. Use 'all' to run all functions. Available functions: {', '.join(GEN_FUNCTIONS.keys())}",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip generating schedules that already exist.",
    )
    args = parser.parse_args()

    if "all" in args.functions:
        args.functions = list(GEN_FUNCTIONS.keys())

    # Run the specified functions
    for func_name in args.functions:
        inner_dir = args.path / func_name
        inner_dir.mkdir(parents=True, exist_ok=True)

        func = GEN_FUNCTIONS.get(func_name, None)

        extra_kwargs = {}
        if args.height is not None:
            extra_kwargs["height"] = args.height
        if args.width is not None:
            extra_kwargs["width"] = args.width
        if args.guidance_scale is not None:
            extra_kwargs["guidance_scale"] = args.guidance_scale

        if func is None:
            print(f"Function {func_name} not recognized.")
            continue

        print(f"Generating {func_name} schedules.")
        schedules = func(
            num_blocks=args.num_blocks,
            num_single_blocks=args.num_single_blocks,
            num_inference_steps=args.num_inference_steps,
            **extra_kwargs,
        )
        save_schedules(schedules, inner_dir, args.skip_existing)


if __name__ == "__main__":
    main()
