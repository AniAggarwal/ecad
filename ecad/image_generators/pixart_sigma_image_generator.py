from pathlib import Path

from ecad.image_generators.pixart_image_generator import (
    PixArtImageGenerator,
)


class PixArtSigmaImageGenerator(PixArtImageGenerator):
    """A wrapper class for generating images using the edited PixArtSigma pipeline.

    Attributes:
        weights_name (str): The name of huggingface weights to use for the transformer in `from_pretrained` format.
            Note that the rest of the pipeline weights are always loaded from PixArt-alpha/PixArt-Sigma-XL-2-1024-MS.
        device (torch.device): The device to use for inference.
        pipeline (PixArtSigmaPipeline): The PixArtSigma pipeline used for generating images.
    """

    DEFAULT_WEIGHTS = "PixArt-alpha/PixArt-Sigma-XL-2-256x256"
    PIPELINE_WEIGHTS = "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS"
    DEFAULT_PIPELINE_NAME = "pixart_sigma"

    def __init__(
        self,
        weights_name: str = DEFAULT_WEIGHTS,
        start_seed: int = 0,
        seed_step: int = 1,
        schedule_path: Path | None = None,
        pipeline_weights: str = PIPELINE_WEIGHTS,
        pipeline_name: str = DEFAULT_PIPELINE_NAME,
    ):
        super().__init__(
            weights_name=weights_name,
            start_seed=start_seed,
            seed_step=seed_step,
            schedule_path=schedule_path,
            pipeline_weights=pipeline_weights,
            pipeline_name=pipeline_name,
        )
