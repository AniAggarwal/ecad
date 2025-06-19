# Evolutionary Caching to Accelerate Your Off-the-Shelf Diffusion Model

This repository contains the official implementation of ECAD (Evolutionary Caching for Accelerated Diffusion), a method that uses evolutionary algorithms to optimize caching schedules for diffusion models. ECAD automatically discovers optimal layer-wise caching patterns that accelerate inference while maintaining image quality. Our approach works with off-the-shelf diffusion models including PixArt-α, PixArt-Σ, and FLUX, achieving significant speedups without requiring model retraining or architectural modifications.

[**Paper Link**](https://arxiv.org/abs/2506.15682) *(Submission Under Review)*  
[**Project Website Link**](https://aniaggarwal.github.io/ecad/)

## Requirements

Follow these steps to set up the environment and install all necessary dependencies:

1. **Create a virtual environment**

   We use Conda with the provided `ecad.yml` to set up our environment, but you're free to use Python's built-in `venv` or any other environment manager you prefer.

2. **Create the Conda environment** using the provided environment file:

   ```bash
   conda env create -f ecad.yml
   ```

3. **Activate the environment**:

   ```bash
   conda activate ecad
   ```

4. **Install Python dependencies** via pip:

   ```bash
   pip install -r requirements.txt
   ```

> Ensure that you are using Python 3.10 or later as specified in `ecad.yml`.

5. **Install the package in editable mode**:

   ```bash
   pip install -e .
   ```

   This installs the `ecad` directory as a package in 'edit' mode. If you run into issues with relative imports, rerun this command from the git root.

6. **Configure environment variables**:

   Copy the `.env_template` file as follows:

   ```bash
   cp .env_template .env
   ```

   Then modify the `.env` file to match your local setup. Update any paths accordingly. For example:

   ```bash
   # In .env
   PROJECT_ROOT="~/ecad" # SET THIS TO YOUR PROJECT ROOT, i.e. where this file is located, as returned by `git rev-parse --show-toplevel`
   ```

   Replace the example path with the actual location of your project directory on your system.

### Preventing OOM Issues
When running any of the scripts, make sure to have set the below variable; alternatively add this to your shell rc file.
This will prevent OOM issues during image generation.

```bash
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
```

## ECAD Optimization

ECAD optimization is achieved by running the `train_nsga2_single_gpu.py` script, which implements the NSGA‑II algorithm for single GPU execution. This script runs the optimization process sequentially, executing image generation, scoring, and metrics computation steps one after another.

We **recommend** using `nohup` when running on a remote server to protect against SSH connectivity issues.

Below is an example command to optimize the PixArt-α model at 256×256 resolution. Results (all schedules by generation) will be saved under `results/genetic/pixart_alpha/pixart_alpha_256_random_init` and all benchmark results will be in `results/benchmark/genetic/pixart_alpha/pixart_alpha_256_random_init`:

```bash
nohup python ecad/genetic/train_nsga2_single_gpu.py \
    --image-generator PixArtAlphaImageGenerator \
    --name pixart_alpha_256_random_init \
    --num-cycles 50 \
    --all-populations-dir results/genetic/pixart_alpha \
    --all-benchmarks-dir results/benchmark/genetic/pixart_alpha \
    --batch-size 100 \
    --population-size 72 \
    --embedding-dir /path/to/results/embeddings/image_reward/pixart_alpha_embeddings \
    --benchmark-prompts prompts/ImageRewardPrompts.json \
    &> nohup_pixart_alpha_ecad_optimize.out & disown
```

**Note**: When starting a new optimization run without an existing population, you will be prompted to confirm random initialization.

To **resume** optimization from a previous checkpoint, replace the `--name` flag with:

```bash
--load-from results/genetic/pixart_alpha/pixart_alpha_256_random_init/gen_010/manager_config.json
```

### Using an Initial Population

To use a custom initial population for optimization instead of random initialization:

1. **Prepare your population directory structure**:
   - Choose a population name (e.g., `initialized_population`)
   - Create the directory structure: `results/genetic/pixart_alpha/initialized_population/gen_000/candidates/`
   - Place your schedule JSON files in the `candidates` directory
   - Name them as `cand_000.json`, `cand_001.json`, ..., `cand_071.json` (for a population size of 72)

2. **Run optimization with your initial population**:

```bash
nohup python ecad/genetic/train_nsga2_single_gpu.py \
    --image-generator PixArtAlphaImageGenerator \
    --name initialized_population \
    --num-cycles 50 \
    --all-populations-dir results/genetic/pixart_alpha \
    --all-benchmarks-dir results/benchmark/genetic/pixart_alpha \
    --batch-size 100 \
    --population-size 72 \
    --embedding-dir /path/to/results/embeddings/image_reward/pixart_alpha_embeddings \
    --benchmark-prompts prompts/ImageRewardPrompts.json \
    &> nohup_pixart_alpha_ecad_optimize.out & disown
```

**Important**: When using an initial population, you will NOT receive a prompt asking about random initialization. The system will automatically detect and use your provided population.

The single GPU implementation executes the following steps sequentially for each generation:
1. Image generation using the evolved schedules
2. Image quality scoring
3. Computational metrics calculation (MACs/FLOPs)

To see the outputs of each evaluation step during execution, add the `--print-eval-outputs` flag.

For a full list of options and parameters, simply append the `--help` flag:

```bash
python ecad/genetic/train_nsga2_single_gpu.py --help
```

## ECAD Schedules

All caching schedules are stored as JSON files that model the binary tensor format detailed in the paper. The schedules referenced in the paper can be found under:

```bash
schedules/schedules_in_paper
```

Additionally, the set of schedules used to initilize the populations for each ECAD optimization of PixArt-α, PixArt-Σ, and FLUX-1.dev are included here:
```bash
schedules/population_initialization/
```

When generating images, you will need to provide a path to a schedule file, or a whole directory of schedules for each of which images will be generated.

Note that resolution, guidance scale, custom weights, and custom pipelines are all managed through these JSON files.

Some schedules used to create heuristics are also included in the `schedules` directory.

## Preparing Prompts

All prompt files used in the paper are available in the `prompts/` directory:

* `COCO_caption_prompts_30k.txt`
* `DrawBench200.txt`
* `ImageRewardPrompts.txt`
* `PartiPrompts.txt`
* `MJHQ-30K-prompts.json`
* `samples.txt` (provided by the PixArt-α authors, unmodified; may contain typos)

We include prompts to generate images included in qualitative figures of the paper in this directory too.

Before running inference, generate and save embeddings for your desired prompts using `generate_embeddings.py`. In these examples, embeddings are saved under `results/embeddings/<dataset>/<model_name>`.

**Important**: Embeddings are model-specific and cannot be shared between different model types. Always generate separate embeddings for each model you plan to use.

For example, for ImageReward prompts and PixArt-α:

```bash
python ecad/benchmark/generate_embeddings.py \
    --image-generator PixArtAlphaImageGenerator \
    --file-path prompts/ImageRewardPrompts.txt \
    --output-dir results/embeddings/image_reward/pixart_alpha_embeddings \
    --batch-size 32
```

This command processes the prompts in `prompts/ImageRewardPrompts.txt` with the specified image generator, saving embedding files to `results/embeddings/image_reward/pixart_alpha_embeddings`. Adjust `--batch-size` as needed.

For detailed usage, run:

```bash
python ecad/benchmark/generate_embeddings.py --help
```

## Image Generation

Once embeddings are prepared, use `generate_images.py` to render images under each caching schedule. In these examples, output images are saved under `results/benchmark/<dataset>/<model_name>` with subdirectories per schedule.

For example, for ImageReward and PixArt-α:

```bash
python ecad/benchmark/generate_images.py \
    --image-generator PixArtAlphaImageGenerator \
    --input-dir results/embeddings/image_reward/pixart_alpha_embeddings \
    --schedule-dir schedules/schedules_in_paper \
    --output-dir results/benchmark/image_reward/pixart_alpha \
    --num-images-per-prompt 10 \
    --batch-size 100 \
    --seed 0 \
    --seed-step 1
```

This will process each prompt embedding in `results/embeddings/image_reward/pixart_alpha_embeddings`, apply every JSON schedule found in `schedules/schedules_in_paper`, and save the results into `results/benchmark/image_reward/pixart_alpha/<schedule-name>/`. Adjust `--num-images-per-prompt`, `--batch-size`, or other flags as needed.

For full parameter details, run:

```bash
python ecad/benchmark/generate_images.py --help
```

## Inference

For quick inference without pre-generating embeddings, use the `inference.py` script. This tool allows you to generate images directly from prompts or prompt files, automatically handling embedding generation and image creation in one step.

### Single Prompt Inference

Generate images from a single text prompt:

```bash
python ecad/inference/inference.py \
    PixArtAlphaImageGenerator \
    --prompt "A fremen stands atop a towering sand dune, clad in a stillsuit and gripping a worm hook in each hand, facing the vast, sun-scorched desert." \
    --num-images-per-prompt 5 \
    --output-dir results/inference/single_prompt
```

### Batch Inference from File

Generate images for multiple prompts from a text file:

```bash
python ecad/inference/inference.py \
    PixArtAlphaImageGenerator \
    --prompt-file prompts/my_prompts.txt \
    --num-images-per-prompt 10 \
    --batch-size-generate 4 \
    --output-dir results/inference/batch_prompts
```

### Using Caching Schedules

Apply a specific caching schedule during inference:

```bash
python ecad/inference/inference.py \
    PixArtAlphaImageGenerator \
    --schedule schedules/schedules_in_paper/ours_fast.json \
    --prompt "A futuristic cityscape with flying cars" \
    --num-images-per-prompt 5
```

### Reusing Pre-generated Embeddings

You can reuse embeddings generated from previous runs for faster inference:

```bash
python ecad/inference/inference.py \
    PixArtAlphaImageGenerator \
    --input-embeddings results/inference/previous_run/embeddings \
    --schedule schedules/schedules_in_paper/ours_fast.json \
    --num-images-per-prompt 20 \
    --output-dir results/inference/from_embeddings
```

**Important**: Embeddings are model-specific and should NOT be shared across different models. Always use embeddings generated by the same model type for inference.

### Advanced Options

Override schedule parameters for custom resolutions or guidance:

```bash
python ecad/inference/inference.py \
    FluxImageGenerator \
    --schedule schedules/schedules_in_paper/flux_schedule.json \
    --prompt-file prompts/creative_prompts.txt \
    --height 768 \
    --width 768 \
    --guidance-scale 7.5 \
    --start-seed 42 \
    --seed-step 1
```

The script automatically:
- Generates and saves embeddings to `<output-dir>/embeddings/` for future reuse
- Saves generated images to `<output-dir>/images/`
- Handles batch processing efficiently

**Note**: For working with pre-generated embeddings (e.g., from benchmark datasets), refer to the Image Generation section above. For PixArt models at resolutions other than 256×256, use a custom schedule that specifies the appropriate height, width, and transformer weights.

For complete usage details:

```bash
python ecad/inference/inference.py --help
```

## Benchmarking

### Computing MACs

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

### Latency Computation

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

### Image Reward Scoring

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

### FID Computation

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

### CLIP Score Computation

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

## COCO

First, generate prompt embeddings using the specialized COCO script. This script reads the entire COCO caption file in megabatches of 3000 lines, encodes each prompt with the specified image generator, and saves the resulting vectors into subfolders under your output directory.

```bash
python ecad/benchmark/generate_coco_embeddings.py \
    --image-generator PixArtAlphaImageGenerator \  
    --file-path prompts/COCO_caption_prompts_30k.txt \  
    --output-dir results/embeddings/coco/pixart_alpha_embeddings \  
    --prompt-batch-size 1000 \  
    --seed 0
```

* `--image-generator`: name of the registered generator (e.g., `PixArtAlphaImageGenerator`).
* `--file-path`: path to the COCO captions text file (one prompt per line).
* `--output-dir`: base directory where embeddings are written, organized in `megabatch_0`, `megabatch_1`, … subfolders.
* `--prompt-batch-size`: number of prompts to encode at once (adjust to fit GPU memory).
* `--seed`: initial random seed and increment between prompts. 

Once embeddings are available under `results/embeddings/coco/pixart_alpha_embeddings`, generate images for each caching schedule by pointing `generate_images.py` at the embeddings root. Output images will be saved under `results/benchmark/coco/pixart_alpha`, with one folder per schedule.

```bash
python ecad/benchmark/generate_images.py \
    --image-generator PixArtAlphaImageGenerator \  
    --input-dir results/embeddings/coco/pixart_alpha_embeddings \  
    --schedule-dir schedules/schedules_in_paper \  
    --output-dir results/benchmark/coco/pixart_alpha \  
    --num-images-per-prompt 10 \  
    --batch-size 100 \  
    --seed 0 \  
    --seed-step 1
```

* `--input-dir`: path to COCO embeddings directory.
* The other flags match the standard `generate_images.py` usage: specify your schedules folder, output location, number of images per prompt, batch size, and seeds.

Optionally, you can split large prompt sets into smaller files and run multiple `generate_images.py` processes in parallel across GPUs.

After image generation, measure FID and CLIP scores exactly as with other datasets. For COCO FID, either supply a local copy of the COCO reference images or use precomputed stats via [clean-fid](https://github.com/GaParmar/clean-fid). For example:

```bash
python ecad/benchmark/compute_fid.py \
    --ref-dir /path/to/coco/ref/images \  
    --gen-dir results/benchmark/coco/pixart_alpha \  
    --model-name inception_v3
```

## MJHQ

Use the dedicated MJHQ script to handle category‑based JSON prompts. This script reads the MJHQ JSON file, groups prompts by their `category` field, batches large prompt sets to avoid OOM, and outputs embeddings into category subfolders under your chosen directory.

```bash
python ecad/benchmark/generate_mjhq_embeddings.py \
    --image-generator PixArtAlphaImageGenerator \
    --file-path prompts/MJHQ-30K-prompts.json \
    --output-dir results/embeddings/mjhq/pixart_alpha_embeddings \
    --prompt-batch-size 1000 \
    --seed 0 
```

* `--image-generator`: registered image generator (e.g., `PixArtAlphaImageGenerator`).
* `--file-path`: path to `MJHQ-30K-prompts.json`.
* `--output-dir`: base directory; embeddings will be saved under `results/embeddings/mjhq/pixart_alpha_embeddings/<category>`.
* `--prompt-batch-size`: batch size for encoding to prevent GPU OOM.
* `--seed`: control random seeds for reproducibility.

When complete, embeddings reside in `results/embeddings/mjhq/pixart_alpha_embeddings`. Next, run the usual image generation step—`generate_images.py`—pointing at the MJHQ embeddings root. Outputs go under `results/benchmark/mjhq/pixart_alpha`:

```bash
python ecad/benchmark/generate_images.py \
    --image-generator PixArtAlphaImageGenerator \
    --input-dir results/embeddings/mjhq/pixart_alpha_embeddings \
    --schedule-dir schedules/schedules_in_paper \
    --output-dir results/benchmark/mjhq/pixart_alpha \
    --num-images-per-prompt 10 \
    --batch-size 100 \
    --seed 0 \
    --seed-step 1
```

* `--input-dir`: path to MJHQ embeddings.
* All other flags match standard `generate_images.py` invocation.

Finally, compute CLIP and FID metrics with MJHQ‑specific flags. For CLIP:

```bash
python ecad/benchmark/compute_clip.py \
    --prompts-path prompts/MJHQ-30K-prompts.json \
    --mjhq \
    --image-dir results/benchmark/mjhq/pixart_alpha/optimized_schedule \
    --file-mode json \
    --clip-model ViT-B/32 \
    --image-naming-mode mjhq
```

* `--mjhq`: indicates MJHQ prompt format.
* `--prompts-path`: the original JSON prompt file.
* `--image-dir`: root of MJHQ-generated images.

For FID, ensure you have the MJHQ reference images downloaded from [Hugging Face](https://huggingface.co/datasets/playgroundai/MJHQ-30K), then run:

```bash
python ecad/benchmark/compute_fid.py \
    --ref-dir <path_to_mjhq_ref_dir> \
    --gen-dir results/benchmark/mjhq/pixart_alpha/optimized_schedule \
    --model-name inception_v3
```

## Contact

For issues, questions, or contributions:
- **GitHub Issues**: Please open an issue on this [GitHub repository](https://github.com/AniAggarwal/ecad/issues) for bug reports or feature requests
- **Email**: For other inquiries, contact Anirud Aggarwal at anirud@umd.edu
