# Benchmarking

## Computing MACs

After generating images, you can compute static complexity metrics—such as Multiply–Accumulate operations (MACs) and FLOPs—for each caching schedule using the `compute_macs.py` script.

Example command (using ImageReward and PixArt-α schedules):

```bash
python ecad/benchmark/compute_macs.py \
    --image-generator PixArtAlphaImageGenerator \
    --input-dir schedules/schedules_in_paper \
```

This script reads each JSON schedule and dumps computed MACs/FLOPs into the schedule file itself.

Use `--recompute-existing` to force re‑calculation of metrics for schedules that already have results.

For full usage information, run:

```bash
python ecad/benchmark/compute_macs.py --help
```

## Latency Computation

To measure the actual runtime latency of each caching schedule on your prompt embeddings, use the `compute_latency.py` script. This script runs inference repeatedly to warm up the GPU and averages over multiple batches. Results are saved directly into the schedule JSON file directly.

Example command (ImageReward + PixArt-α):

```bash
python ecad/benchmark/compute_latency.py \
    --image-generator PixArtAlphaImageGenerator \
    --embedding-dir results/embeddings/image_reward/pixart_alpha_embeddings \
    --schedule-dir schedules/schedules_in_paper \
    --batch-size 100 \
    --warmup-steps 1 \
    --num-samples 5
```

* `--warmup-steps` and `--num-samples` should follow the values reported in the paper's appendix to match the benchmarking protocol.
* You can also specify `--schedule-list` to measure a specific subset of schedule files.
* Use `--recompute-existing` to force re‑evaluation of latency for schedules with prior results.

For full usage details, run:

```bash
python ecad/benchmark/compute_latency.py --help
```

## Image Reward Scoring

After computing latency, evaluate the quality of your generated images using the ImageReward model. The `score_images.py` script processes images in a schedule-specific directory and attaches a score for each prompt.

Example command (ImageReward + PixArt-α):

```bash
python ecad/benchmark/score_images.py \
    --benchmark-prompts prompts/ImageRewardPrompts.json \
    --image-dir results/benchmark/image_reward/pixart_alpha/optimized_schedule \
    --output-subpath results/benchmark/image_reward/pixart_alpha/scores/image_reward_scores.json \
    --image-naming-mode image_reward \
    --file-mode json \
    --exactly-n-images 100
```

* `--benchmark-prompts`: JSON file of prompts (e.g., `prompts/MJHQ-30K-prompts.json`) or text file with one prompt per line when paired with `--file-mode txt`.
* `--image-dir`: Root directory containing images or subfolders of images per schedule.
* `--output-subpath`: Destination path for the resulting score JSON relative to the `--image-dir`.
* `--image-naming-mode`: Choose from `image_reward`, `parti`, or `toca` to match how images are named. Detailed regex can be found in the code, though it is generally safe to just use `parti`.
* `--file-mode`: `json`, `text`, `txt`, or `tsv` to match your prompt file format.
* `--exactly-n-images`: Only score directories containing exactly *n* images (prevents partial runs).
* Use `--delete-after` to remove images post-scoring and `--rescore-existing` to overwrite prior results.

For complete details, run:

```bash
python ecad/benchmark/score_images.py --help
```

## FID Computation

To assess the similarity between generated and reference images, compute the Fréchet Inception Distance (FID) using `compute_fid.py`. This measures the distributional distance between real and generated image features.

Example command (ImageReward + PixArt-α):

```bash
python ecad/benchmark/compute_fid.py \
    --ref-dir data/MJHQ-30K \
    --gen-dir results/benchmark/image_reward/pixart_alpha/ours_fast \
    --model-name inception_v3
```

* `--ref-dir`: Path to the reference MJHQ‑30K image directory. Note the directory structures must match between the reference and generated images directories.
* `--gen-dir`: Path to your generated images directory for a given schedule.
* `--model-name`: Inception variant to use (default: `inception_v3`).

For full usage details, run:

```bash
python ecad/benchmark/compute_fid.py --help
```

## CLIP Score Computation

Evaluate semantic alignment between your prompts and images using CLIP via `compute_clip.py`.

Example command (ImageReward + PixArt-α):

```bash
python ecad/benchmark/compute_clip.py \
    --prompts-path prompts/ImageRewardPrompts.json \
    --image-dir results/benchmark/image_reward/pixart_alpha/ours_fast \
    --file-mode json \
    --clip-model ViT-B/32 \
    --image-naming-mode image_reward
```

* `--prompts-path`: File containing the original prompts (JSON, text, or TSV).
* `--image-dir`: Directory with generated images to score.
* `--file-mode`: Format of the prompt file (`json`, `text`, `tsv`).
* `--clip-model`: Name of the CLIP model to use (e.g., `ViT-B/32`).
* `--image-naming-mode`: Pattern for matching image filenames (e.g., `image_reward`, `parti`, `toca`, `mjhq`, `coco`).

For full usage details, run:

```bash
python ecad/benchmark/compute_clip.py --help
```