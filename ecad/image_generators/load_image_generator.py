from ecad.image_generators.flux_image_generator import (
    FluxImageGenerator,
)
from ecad.image_generators.image_generator import (
    ImageGenerator,
)
from ecad.image_generators.pixart_alpha_image_generator import (
    PixArtAlphaImageGenerator,
)
from ecad.image_generators.pixart_sigma_image_generator import (
    PixArtSigmaImageGenerator,
)
from ecad.types import ImageGeneratorConfig


class ImageGeneratorRegistry:
    registry: dict[str, type[ImageGenerator]] = {
        "PixArtAlphaImageGenerator": PixArtAlphaImageGenerator,
        "PixArtSigmaImageGenerator": PixArtSigmaImageGenerator,
        "FluxImageGenerator": FluxImageGenerator,
    }

    @classmethod
    def get(
        cls, name: str, default_name: str | None = None
    ) -> type[ImageGenerator] | None:
        """Get the image generator class by name, or the default image generator class if not found.

        Args:
            name: the name the image generator was registered with.
            default_name: the default image generator name to return if the named image generator is not found.

        Returns:
            The image generator class.
        """
        image_generator = cls.registry.get(name, None)
        if image_generator is None and default_name is not None:
            image_generator = cls.registry.get(default_name, None)

        return image_generator


def get_image_generator_type_from_config(
    config: ImageGeneratorConfig, default_name: str = "PixArtImageGenerator"
) -> type[ImageGenerator]:
    """Get the image generator class by name.

    Args:
        config: the image generator configuration.
        default_name: the default image generator name.

    Returns:
        The image generator class.
    """
    if "image_generator" not in config:
        img_gen = None
    else:
        img_gen = ImageGeneratorRegistry.get(
            config["image_generator"], default_name
        )

    if img_gen is None:
        raise ValueError(f"Image generator not found in config: {config}.")

    return img_gen


def get_image_generator_type(
    name: str, default_name: str = "PixArtImageGenerator"
) -> type[ImageGenerator]:
    """Get the image generator class by name.

    Args:
        name: the name of the image generator.
        default_name: the default image generator name.

    Returns:
        The image generator class.
    """
    img_gen = ImageGeneratorRegistry.get(name, default_name)
    if img_gen is None:
        raise ValueError(f"Image generator not found: {name}.")

    return img_gen
