# Datasets

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