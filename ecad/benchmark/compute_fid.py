import argparse
import json
from pathlib import Path
from cleanfid import fid

MJHQ_CUSTOM_STATS_NAME = "mjhq-30k"


def score_fid(
    ref_dir: Path, gen_dir: Path, output_subpath: Path, model_name: str
) -> None:
    ref_dir_str = str(ref_dir.resolve())
    gen_dir_str = str(gen_dir.resolve())

    # generate FID for ref dir, if necessary
    if not fid.test_stats_exists(
        MJHQ_CUSTOM_STATS_NAME, mode="clean", model_name=model_name
    ):
        print(
            f"No custom stats found for {MJHQ_CUSTOM_STATS_NAME}. Generating..."
        )
        fid.make_custom_stats(
            MJHQ_CUSTOM_STATS_NAME,
            ref_dir_str,
            mode="clean",
            model_name=model_name,
        )

    print("Computing FID for generated images...")
    score = fid.compute_fid(
        gen_dir_str,
        dataset_name=MJHQ_CUSTOM_STATS_NAME,
        mode="clean",
        dataset_split="custom",
        model_name=model_name,
        num_workers=4,
    )
    print(f"FID score: {score}")

    output = {
        "ref_dir": ref_dir_str,
        "gen_dir": gen_dir_str,
        "model_name": model_name,
        "fid_score": score,
    }

    fid_output_path = gen_dir / output_subpath
    with open(fid_output_path, "w") as f:
        json.dump(output, f, indent=4)
    print(f"FID output saved to {fid_output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Computes the FID scores for the generated images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--ref-dir",
        type=Path,
        required=True,
        help="Path to the refrence MJHQ-30K dir.",
    )
    parser.add_argument(
        "--gen-dir",
        type=Path,
        required=True,
        help="Path to the generated images.",
    )
    parser.add_argument(
        "--output-subpath",
        "-o",
        type=Path,
        default=Path("fid_scores.json"),
        help="Filename or subpath for the generated FID JSON file.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="inception_v3",
        help="Name of the model to use for FID computation.",
    )
    args = parser.parse_args()

    if not args.gen_dir.exists() or not args.gen_dir.is_dir():
        print(f"Generated images dir {args.gen_dir} not found.")
        return

    score_fid(args.ref_dir, args.gen_dir, args.output_subpath, args.model_name)


if __name__ == "__main__":
    main()
