# Using an Initial Population

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