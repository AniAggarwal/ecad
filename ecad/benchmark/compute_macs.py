import argparse
import json
import gc
from pathlib import Path
from typing import Any

import torch
from calflops import calculate_flops

from ecad.image_generators.image_generator import ImageGenerator
from ecad.image_generators.flux_image_generator import (
    FluxImageGenerator,
)
from ecad.image_generators.load_image_generator import (
    ImageGeneratorRegistry,
    get_image_generator_type,
)
from ecad.image_generators.pixart_image_generator import (
    PixArtImageGenerator,
)

WEIGHTS_TO_SHAPES = {
    "PixArt-alpha/PixArt-XL-2-256x256": {
        "height_width_multiplier": 1,
        "resolution": None,
        "aspect_ratio": None,
    },
    "PixArt-alpha/PixArt-Sigma-XL-2-256x256": {
        "height_width_multiplier": 1,
        "resolution": None,
        "aspect_ratio": None,
    },
    "PixArt-alpha/PixArt-XL-2-1024-MS": {
        "height_width_multiplier": 4,
        "resolution": [1024, 1024],
        "aspect_ratio": [1],
    },
}

BATCH_SIZE = 2  # default to 2 since we get text and null embedding
CHANNELS = 4
SEQ_LEN = 120
EMBED_DIM = 4096


def create_inputs(image_generator: ImageGenerator) -> dict[str, Any]:
    if isinstance(image_generator, PixArtImageGenerator):
        return create_inputs_pixart(
            image_generator.transformer_weights,
        )
    elif isinstance(image_generator, FluxImageGenerator):
        return create_inputs_flux(
            image_generator.height,
            image_generator.width,
        )
    else:
        raise ValueError("Unsupported image generator type.")


def create_inputs_pixart(
    weights: str,
    batch_size: int = BATCH_SIZE,
    channels: int = CHANNELS,
) -> dict[str, Any]:
    device = torch.device("cpu")
    dtype = torch.half

    weight_shapes = WEIGHTS_TO_SHAPES.get(weights)
    if weight_shapes is None:
        raise ValueError(f"Shapes for weights {weights} not found.")

    hw_multi = weight_shapes.get("height_width_multiplier", 1)
    hw = 32 * hw_multi
    resolution = weight_shapes.get("resolution")
    aspect_ratio = weight_shapes.get("aspect_ratio")

    if resolution is not None:
        res_input = torch.tensor(
            [resolution] * batch_size, device="cuda", dtype=dtype
        )
    else:
        res_input = None
    if aspect_ratio is not None:
        ar_input = torch.tensor(
            aspect_ratio * batch_size, device="cuda", dtype=dtype
        )
    else:
        ar_input = None

    return {
        "hidden_states": torch.zeros(
            batch_size, channels, hw, hw, device=device, dtype=dtype
        ),
        "encoder_hidden_states": torch.zeros(
            batch_size, SEQ_LEN, EMBED_DIM, device=device, dtype=dtype
        ),
        "timestep": torch.zeros(batch_size, device=device, dtype=dtype),
        "attention_mask": None,
        "encoder_attention_mask": torch.zeros(
            batch_size, SEQ_LEN, device=device, dtype=dtype
        ),
        "added_cond_kwargs": {
            "resolution": res_input,
            "aspect_ratio": ar_input,
        },
        "cross_attention_kwargs": None,
        "return_dict": False,
    }


def create_inputs_flux(
    height: int,
    width: int,
    batch_size: int = BATCH_SIZE,
) -> dict[str, Any]:
    device = torch.device("cpu")
    dtype = torch.bfloat16

    if (height % 256 != 0) or (width % 256 != 0) or (height != width):
        raise ValueError(
            "Height and width must be mutliple of 256 and same size."
        )

    hidden_size = 256 * (height // 256) * (width // 256)

    return {
        "hidden_states": torch.zeros(
            batch_size, hidden_size, 64, device=device, dtype=dtype
        ),
        "encoder_hidden_states": torch.zeros(
            batch_size, 512, 4096, device=device, dtype=dtype
        ),
        "pooled_projections": torch.zeros(
            batch_size, 768, device=device, dtype=dtype
        ),
        "timestep": torch.zeros(batch_size, device=device, dtype=dtype),
        "img_ids": torch.zeros(
            batch_size, hidden_size, 3, device=device, dtype=dtype
        ),
        "txt_ids": torch.zeros(batch_size, 512, 3, device=device, dtype=dtype),
        "guidance": torch.zeros(batch_size, device=device, dtype=dtype),
        "joint_attention_kwargs": None,
        "return_dict": False,
    }


def compute_metrics_for_dir(
    image_generator_type: type[ImageGenerator],
    input_dir: Path,
    recompute_existing: bool,
    print_detailed: bool,
) -> None:
    print(f"Computing metrics for {input_dir}")

    if input_dir.is_file():
        compute_metrics_for_schedule(
            image_generator_type, input_dir, recompute_existing, print_detailed
        )
    else:
        for path in input_dir.glob("**/*.json"):
            compute_metrics_for_schedule(
                image_generator_type,
                path,
                recompute_existing,
                print_detailed,
            )
    print("Done.")


def compute_metrics_for_schedule(
    image_generator_type: type[ImageGenerator],
    schedule_path: Path,
    recompute_existing: bool,
    print_detailed: bool,
) -> None:
    with open(schedule_path) as f:
        data = json.load(f)

    if not recompute_existing and data.get("metrics", {}).get(
        "total_macs", None
    ):
        print(f"Metrics already computed for {schedule_path}. Skipping.")
        return

    dit_steps = data.get("dit_schedule", {}).get("num_inference_steps")
    cache_steps = data.get("cache_schedule", {}).get("num_inference_steps")

    if dit_steps is None and cache_steps is None:
        raise ValueError(
            "num_inference_steps not found in dit_schedule or cache_schedule"
        )
    if (
        dit_steps != cache_steps
        and dit_steps is not None
        and cache_steps is not None
    ):
        raise ValueError(
            "num_inference_steps mismatch across dit and cache schedules"
        )
    num_inference_steps = dit_steps or cache_steps

    # don't init transformer block for FluxImageGenerator bc it is very slow
    extra_kwargs = {}
    if image_generator_type == FluxImageGenerator:
        extra_kwargs["skip_transformer_block_init"] = True

    image_gen = image_generator_type(schedule_path=schedule_path)
    image_gen.create_diffusion_pipeline(**extra_kwargs)

    # save VRAM by moving everything but the transformer to CPU
    image_gen.diffusion_pipeline = image_gen.diffusion_pipeline.to("cpu")
    image_gen.diffusion_pipeline.transformer = (
        image_gen.diffusion_pipeline.transformer.to("cuda")
    )

    metrics = compute_metrics_by_pipeline(
        image_gen, num_inference_steps, print_detailed
    )
    image_gen.free_diffusion_pipeline()
    del image_gen
    gc.collect()
    torch.cuda.empty_cache()

    # Re-open file in case it changed.
    with open(schedule_path, "r") as f:
        data = json.load(f)
    old_metrics = data.pop("metrics", {})

    # only overwrite the common keys
    for k, v in metrics.items():
        old_metrics[k] = v

    data["metrics"] = old_metrics

    with open(schedule_path, "w") as f:
        json.dump(data, f, indent=4)

    print(f"Finished computing metrics for {schedule_path}")


def compute_metrics_by_pipeline(
    image_gen: ImageGenerator,
    num_inference_steps: int,
    print_detailed: bool,
) -> dict[str, Any]:
    if "pipeline" in image_gen.config:
        if image_gen.config["pipeline"]["name"] == "tgate":
            return compute_metrics_tgate(
                image_gen, num_inference_steps, print_detailed
            )

    return compute_metrics(image_gen, num_inference_steps, print_detailed)


@torch.no_grad()
def compute_metrics(
    image_gen: ImageGenerator,
    num_inference_steps: int,
    print_detailed: bool,
) -> dict[str, Any]:
    if image_gen.diffusion_pipeline is None:
        raise ValueError("Diffusion pipeline not found/instantiated.")

    by_inference_step = {}
    total_flops = 0
    total_macs = 0
    for step in range(num_inference_steps):
        assert (step == image_gen.dit_scheduler.curr_step) and (
            image_gen.dit_scheduler.curr_step
            == image_gen.cache_schedule.curr_step
        ), "Step mismatch!"

        sample_input: dict[str, torch.Tensor] = create_inputs(image_gen)

        if print_detailed:
            print("\n" + "=" * 80)
            print("=" * 35 + f" STEP {step:03d} " + "=" * 35)
            print("=" * 80)
        flops, macs, _ = calculate_flops(
            image_gen.diffusion_pipeline.transformer,
            kwargs=sample_input,
            include_backPropagation=False,
            print_results=print_detailed,
            print_detailed=print_detailed,
            output_as_string=False,
        )
        by_inference_step[f"{step:03}"] = {"flops": flops, "macs": macs}
        total_flops += flops  # type: ignore
        total_macs += macs  # type: ignore

        # simulate calling callbacks at the end of each step
        for callback in image_gen.callbacks:
            callback(step, 0)
        # image_gen._call_callbacks(step, 0)

    metrics: dict[str, Any] = {
        "by_inference_step": by_inference_step,
        "total_flops": total_flops,
        "total_flops_T": total_flops / 1000**4,
        "total_macs": total_macs,
        "total_macs_T": total_macs / 1000**4,
    }
    return metrics


def compute_metrics_tgate(
    image_gen: PixArtImageGenerator,
    num_inference_steps: int,
    print_detailed: bool,
) -> dict[str, Any]:
    if image_gen.diffusion_pipeline is None:
        raise ValueError("PixArt pipeline not found/instantiated.")

    try:
        gate_step = image_gen.config["pipeline"]["kwargs"]["gate_step"]
    except KeyError as e:
        raise ValueError(
            "config['pipeline']['kwargs']['gate_step'] are all required for TGATE."
        ) from e

    by_inference_step = {}
    total_flops = 0
    total_macs = 0
    for step in range(num_inference_steps):
        assert (step == image_gen.dit_scheduler.curr_step) and (
            image_gen.dit_scheduler.curr_step
            == image_gen.cache_schedule.curr_step
        ), "Step mismatch!"

        # just null embeddings from gate_step onwards
        batch_size = 2 if step < gate_step else 1
        sample_input: dict[str, torch.Tensor] = create_inputs_pixart(
            image_gen.transformer_weights, batch_size
        )

        if print_detailed:
            print("\n" + "=" * 80)
            print("=" * 35 + f" STEP {step:03d} " + "=")
            print("=" * 80)

        flops, macs, _ = calculate_flops(
            image_gen.diffusion_pipeline.transformer,
            kwargs=sample_input,
            include_backPropagation=False,
            print_results=print_detailed,
            print_detailed=print_detailed,
            output_as_string=False,
        )

        by_inference_step[f"{step:03}"] = {"flops": flops, "macs": macs}
        total_flops += flops  # type: ignore
        total_macs += macs  # type: ignore

        # simulate calling callbacks at the end of each step
        image_gen._call_callbacks(step, 0)

    metrics: dict[str, Any] = {
        "by_inference_step": by_inference_step,
        "total_flops": total_flops,
        "total_flops_T": total_flops / 1000**4,
        "total_macs": total_macs,
        "total_macs_T": total_macs / 1000**4,
    }
    return metrics


def get_compute_macs_argparser() -> argparse.ArgumentParser:
    """
    Returns an ArgumentParser preloaded with compute_macs-specific arguments.
    This lets compute_macs.py be used standalone or merged into a multi-job parser.
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--image-generator",
        type=str,
        required=False,
        help="The name of the image generator to use.",
        choices=list(ImageGeneratorRegistry.registry.keys()),
    )
    parser.add_argument(
        "--input-dir",
        "-i",
        type=Path,
        required=True,
        help="Parent directory (or JSON file) whose schedules will have metrics computed recursively.",
    )
    parser.add_argument(
        "--recompute-existing",
        action="store_true",
        help="Recompute metrics for schedules that already have metric data.",
    )
    parser.add_argument(
        "--print-detailed",
        action="store_true",
        help="Print detailed metrics for each step.",
    )
    return parser


def main():
    # For standalone usage, simply use the compute_macs-specific arguments.
    parser = argparse.ArgumentParser(
        description="Compute MACs, FLOPs, and other static computational complexity metrics for schedules."
    )
    compute_parser = get_compute_macs_argparser()
    # Add compute_macs-specific arguments to the main parser.
    for action in compute_parser._actions:
        parser._add_action(action)
    args = parser.parse_args()

    args.input_dir = args.input_dir.resolve()
    if not args.input_dir.exists():
        raise FileNotFoundError(f"Directory {args.input_dir} not found.")

    image_generator_type = get_image_generator_type(args.image_generator)

    compute_metrics_for_dir(
        image_generator_type,
        args.input_dir,
        args.recompute_existing,
        args.print_detailed,
    )


if __name__ == "__main__":
    main()
