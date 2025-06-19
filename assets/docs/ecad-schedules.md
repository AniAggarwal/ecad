# ECAD Schedules

All caching schedules are stored as JSON files that model the binary tensor format detailed in the paper. The schedules referenced in the paper can be found under:

```bash
schedules/schedules_in_paper
```

Additionally, the set of schedules used to initialize the populations for each ECAD optimization of PixArt-α, PixArt-Σ, and FLUX-1.dev are included here:
```bash
schedules/population_initialization/
```

When generating images, you will need to provide a path to a schedule file, or a whole directory of schedules for each of which images will be generated.

Note that resolution, guidance scale, custom weights, and custom pipelines are all managed through these JSON files.

Some schedules used to create heuristics are also included in the `schedules` directory.