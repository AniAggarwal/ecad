import argparse
from pathlib import Path

from ecad.schedulers.dit_scheduler.generators.pixart_schedule_generators import (
    GEN_FUNCTIONS,
    save_schedules,
)

DEFAULT_SCHEDULE_DIR = Path("schedules/dit_schedules/")
DEFAULT_FUNCTIONS = ["gen_default"]
DEFAULT_NUM_BLOCKS = 28
DEFAULT_NUM_INFERENCE_STEPS = 20


def main():
    parser = argparse.ArgumentParser(
        description="Generate and save transformer schedules."
    )
    parser.add_argument(
        "-p",
        "--path",
        type=Path,
        default=DEFAULT_SCHEDULE_DIR,
        help="Path to save the generated schedules. A directory will be created for the schedules.",
    )
    parser.add_argument(
        "-b",
        "--num_blocks",
        type=int,
        default=DEFAULT_NUM_BLOCKS,
        help="Number of blocks to use in the schedule generation.",
    )
    parser.add_argument(
        "-s",
        "--num_inference_steps",
        type=int,
        default=DEFAULT_NUM_INFERENCE_STEPS,
        help="Number of inference steps to use in the schedule generation.",
    )
    parser.add_argument(
        "-f",
        "--functions",
        nargs="+",
        default=DEFAULT_FUNCTIONS,
        help=f"List of functions to run. Use 'all' to run all functions. Available functions:\n{', '.join(GEN_FUNCTIONS.keys())}",
    )
    parser.add_argument(
        "-v",
        "--visualize",
        action="store_true",
        help="Visualize the schedules. Default: False.",
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

        if func is None:
            print(f"Function {func_name} not recognized.")
            continue

        print(f"Generating {func_name} schedules.")
        schedules = func(
            num_blocks=args.num_blocks,
            num_inference_steps=args.num_inference_steps,
        )
        save_schedules(schedules, inner_dir, args.visualize, args.skip_existing)


if __name__ == "__main__":
    main()
