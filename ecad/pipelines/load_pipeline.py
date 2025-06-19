from typing import Any, Callable
from diffusers.pipelines.pixart_alpha.pipeline_pixart_alpha import (
    PixArtAlphaPipeline,
)
from diffusers.pipelines.pixart_alpha.pipeline_pixart_sigma import (
    PixArtSigmaPipeline,
)
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline

from ecad.pipelines.tgate import TGATEPipeline
from ecad.pipelines.pass_through import PassThroughTransformerPipeline
from ecad.types import PipelineConfig


class PipelineRegistry:
    _registry: dict[str, type[DiffusionPipeline]] = {
        "pixart_alpha": PixArtAlphaPipeline,
        "pixart_sigma": PixArtSigmaPipeline,
        "tgate": TGATEPipeline,
        "flux": FluxPipeline,
        "pass_through": PassThroughTransformerPipeline,
    }

    @classmethod
    def get(
        cls, name: str, default_name: str | None = None
    ) -> type[DiffusionPipeline] | None:
        """Get the pipeline class by name, or the default pipeline class if not found.

        Args:
            name: the name the pipeline was registered with.
            default: the default pipeline name to return if the named pipeline is not found.

        Returns:
            The pipeline class.
        """
        pipeline = cls._registry.get(name, None)
        if pipeline is None and default_name is not None:
            pipeline = cls._registry.get(default_name, None)
        return pipeline


def pipeline_from_pretrained(
    pipeline_config: PipelineConfig, default_pipeline_name: str | None = None
) -> Callable[..., DiffusionPipeline]:
    pipeline_class = PipelineRegistry.get(
        pipeline_config.get("name", default_pipeline_name)
    )
    extra_kwargs = pipeline_config.get("kwargs", {})

    print(f"Using pipeline: {pipeline_class}")
    print(f"\tExtra pipeline kwargs: {extra_kwargs}")

    def from_pretrained(*args: Any, **kwargs: Any) -> DiffusionPipeline:
        return pipeline_class.from_pretrained(*args, **kwargs, **extra_kwargs)  # type: ignore

    return from_pretrained
