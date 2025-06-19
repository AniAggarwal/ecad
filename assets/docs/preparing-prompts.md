# Preparing Prompts

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