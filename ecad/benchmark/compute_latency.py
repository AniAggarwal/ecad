import argparse
import json
from pathlib import Path
import torch

from ecad.image_generators.image_generator import (
    ImageGenerator,
)
from ecad.image_generators.load_image_generator import (
    ImageGeneratorRegistry,
    get_image_generator_type,
)


def get_curr_gpu() -> str:
    idx = torch.cuda.current_device()
    name = torch.cuda.get_device_name(idx)
    return name


def time_for_schedule(
    image_generator_type: type[ImageGenerator],
    schedule_file: Path,
    embedding_dir: Path,
    batch_size: int,
    warmup_steps: int,
    num_samples: int,
    seed: int,
    recompute_existing: bool,
):
    with open(schedule_file) as f:
        data = json.load(f)

    if (
        not recompute_existing
        and data.get("metrics", {}).get("latency", None) is not None
    ):
        print(f"Metrics already computed for {schedule_file}. Skipping.")
        return

    print(f"\n\n\nTiming latency for schedule: {schedule_file.stem}.\n")

    try:
        image_generator = image_generator_type(  # type: ignore
            start_seed=seed, seed_step=0, schedule_path=schedule_file
        )
    except Exception as e:
        raise ValueError(
            "Error creating image generator, check arguments passed during init in the code."
        ) from e

    times = image_generator.time_image_generation(
        embedding_dir,
        batch_size=batch_size,
        num_batches=warmup_steps + num_samples,
        free_after=True,
    )

    warmups = times[:warmup_steps]
    latencies = times[warmup_steps:]
    avg = sum(latencies) / len(latencies)

    output_data = {
        "avg": avg,
        "batch_size": batch_size,
        "num_samples": num_samples,
        "warmup_steps": warmup_steps,
        "gpu": get_curr_gpu(),
        "warmups": warmups,
        "latencies": latencies,
    }

    # reopen just in case it changed
    with open(schedule_file) as f:
        data = json.load(f)

    metrics = data.get("metrics", {})
    metrics.pop("latency", None)
    metrics["latency"] = output_data
    data["metrics"] = metrics

    with open(schedule_file, "w") as f:
        json.dump(data, f, indent=4)

    print(f"Finished and saved metrics to {schedule_file}.")


def time_all_schedules(
    image_generator_type: type[ImageGenerator],
    schedule_dir: Path,
    embedding_dir: Path,
    batch_size: int,
    warmup_steps: int,
    num_samples: int,
    seed: int,
    recompute_existing: bool,
):
    print(f"Timing latencies for {schedule_dir}")

    if schedule_dir.is_file():
        time_for_schedule(
            image_generator_type,
            schedule_dir,
            embedding_dir,
            batch_size,
            warmup_steps,
            num_samples,
            seed,
            recompute_existing,
        )
    else:
        for path in schedule_dir.glob("**/*.json"):
            time_for_schedule(
                image_generator_type,
                path,
                embedding_dir,
                batch_size,
                warmup_steps,
                num_samples,
                seed,
                recompute_existing,
            )

    print("Done.")


def main():
    parser = argparse.ArgumentParser(
        description="Compute the latency of a model on a given schedule."
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
        "--embedding-dir",
        type=Path,
        help="Path to the directory containing prompt embeddings in <name>.pt format.",
        required=True,
    )

    parser.add_argument(
        "-d",
        "--schedule-dir",
        type=Path,
        help="Directory to read schedules from in <schedule-name>.json format.",
    )

    parser.add_argument(
        "--schedule-list",
        nargs="+",
        required=False,
        help="A comma seperated list of paths to <schedule-name>.json files.",
    )

    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=100,
        help=f"Batch size while generating images.",
    )

    parser.add_argument(
        "-w",
        "--warmup-steps",
        type=int,
        default=10,
        help=f"The number of times to run and throw away the first prompt to warm up the GPU.",
    )

    parser.add_argument(
        "-n",
        "--num-samples",
        type=int,
        default=5,
        help=f"The number of batches to run the schedule to average the computed latency.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="The random seed to start each image generation with.",
    )

    parser.add_argument(
        "--recompute-existing",
        action="store_true",
        help="Recompute metrics for schedules that already have metric data.",
    )
    args = parser.parse_args()

    if not args.embedding_dir.exists():
        print(f"Input directory {args.embedding_dir} not found.")
        return

    image_generator_type = get_image_generator_type(args.image_generator)

    if args.schedule_dir is not None:
        if not args.schedule_dir.exists():
            print(f"Directory {args.schedule_dir} for schedules not found.")
            return

        time_all_schedules(
            image_generator_type,
            args.schedule_dir,
            args.embedding_dir,
            args.batch_size,
            args.warmup_steps,
            args.num_samples,
            args.seed,
            args.recompute_existing,
        )

    elif args.schedule_list is not None:
        for schedule in args.schedule_list:
            schedule_path = Path(schedule)
            time_for_schedule(
                image_generator_type,
                schedule_path,
                args.embedding_dir,
                args.batch_size,
                args.warmup_steps,
                args.num_samples,
                args.seed,
                args.recompute_existing,
            )
    else:
        print("No schedule directory or schedule list provided. Exiting.")

    print("Done.")


if __name__ == "__main__":
    main()
