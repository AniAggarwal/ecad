import gc
from pathlib import Path
from typing import Sequence
from abc import ABC

from diffusers.pipelines.pipeline_utils import DiffusionPipeline
import torch
from PIL.Image import Image
from transformers import T5EncoderModel

from ecad.image_generators.image_generator import ImageGenerator
from ecad.schedulers.cache_scheduler.pixart_cache_schedule import (
    PixArtCacheSchedule,
)
from ecad.schedulers.dit_scheduler.pixart_dit_schedule import (
    PixArtDiTSchedule,
)
from ecad.schedulers.dit_scheduler.generators.pixart_schedule_generators import (
    gen_default as dit_gen_default,
)
from ecad.schedulers.cache_scheduler.generators.pixart_schedule_generators import (
    gen_default as cache_gen_default,
)
from ecad.transformer_2d_models.pixart_transformer_2d_edited import (
    PixArtTransformer2DEdited,
)
from ecad.types import ImageGeneratorConfig, PixArtPromptEmbedding


class PixArtImageGenerator(ImageGenerator, ABC):
    """A wrapper class for generating images using the edited PixArt pipeline.

    Attributes:
        weights_name (str): The name of huggingface weights to use for the transformer in `from_pretrained` format.
        device (torch.device): The device to use for inference.
        pipeline (DiffusionPipeline): The PixArt pipeline used for generating images.
    """

    # should be overriden in subclasses
    DEFAULT_WEIGHTS = None
    PIPELINE_WEIGHTS = None
    DEFAULT_PIPELINE_NAME = None
    DEFAULT_HEIGHT = 256
    DEFAULT_WIDTH = 256

    def __init__(
        self,
        weights_name: str | None = None,
        start_seed: int = 0,
        seed_step: int = 1,
        schedule_path: Path | None = None,
        pipeline_weights: str | None = None,
        pipeline_name: str | None = None,
    ):
        if not torch.cuda.is_available():
            raise ValueError("CUDA is required for image generation.")

        weights_name = weights_name or self.DEFAULT_WEIGHTS
        pipeline_weights = pipeline_weights or self.PIPELINE_WEIGHTS
        pipeline_name = pipeline_name or self.DEFAULT_PIPELINE_NAME

        super().__init__(
            default_transformer_weights=weights_name,
            default_pipeline_weights=pipeline_weights,
            default_pipeline_name=pipeline_name,
            schedule_path=schedule_path,
            start_seed=start_seed,
            seed_step=seed_step,
            device="cuda",
            dit_schedule_type=PixArtDiTSchedule,
            cache_schedule_type=PixArtCacheSchedule,
        )

        # PixArt uses the same empty null embeds for negative for all prompts
        self.negative_prompt_embeds: torch.Tensor | None = None
        self.negative_prompt_attention_mask: torch.Tensor | None = None

    def _load_subclass_config_defaults(
        self, config: ImageGeneratorConfig
    ) -> None:
        self.height = config.get("height", self.DEFAULT_HEIGHT)
        self.width = config.get("width", self.DEFAULT_WIDTH)

    def _default_dit_schedule(self) -> PixArtDiTSchedule:
        return next(dit_gen_default(28, 20))

    def _default_cache_schedule(self) -> PixArtCacheSchedule:
        return next(cache_gen_default(28, 20))

    def _call_callbacks_wrapper(
        self,
        step: int,
        timestep: int,
        _latents: torch.Tensor,
    ) -> None:
        self._call_callbacks(step, timestep)

    def create_encoder_pipeline(self) -> None:
        text_encoder = T5EncoderModel.from_pretrained(
            self.pipeline_weights,
            subfolder="text_encoder",
            use_safetensors=True,
        )

        encoder_pipeline: DiffusionPipeline = self.pipeline_from_pretrained(
            self.pipeline_weights,
            text_encoder=text_encoder,
            transformer=None,
            use_safetensors=True,
        ).to(self.device)

        self.encoder_pipeline = encoder_pipeline

        # negative prompt embeddings are not generated when using CFG
        # but we need to generate them for "" for PixArt manually
        with torch.no_grad():
            # must use CFG to generate the negative prompt
            (
                _,
                _,
                self.negative_prompt_embeds,
                self.negative_prompt_attention_mask,
            ) = self.encoder_pipeline.encode_prompt(
                prompt="",
                negative_prompt="",
            )

    def create_diffusion_pipeline(self) -> None:
        """Sets up the PixArt pipeline with a custom PixArt transformer model."""

        print(f"Creating PixArt Transformer with weights {self.transformer_weights}.")
        pixart_transformer = PixArtTransformer2DEdited.from_pretrained(
            self.transformer_weights,
            subfolder="transformer",
            torch_dtype=torch.float16,
            use_safetensors=True,
            dit_scheduler=self.dit_scheduler,
            cache_schedule=self.cache_schedule,
        )

        print(f"Creating PixArt pipeline with weights {self.pipeline_weights}.")
        pipeline: DiffusionPipeline = self.pipeline_from_pretrained(
            self.pipeline_weights,
            transformer=pixart_transformer,
            text_encoder=None,
            torch_dtype=torch.float16,
            use_safetensors=True,
        ).to(self.device)

        self.diffusion_pipeline = pipeline

    @torch.inference_mode()
    def encode_prompts(
        self,
        prompts: list[str],
        negative_prompts: list[str] | None = None,
        batch_size: int | None = None,
        **kwargs,
    ) -> Sequence[PixArtPromptEmbedding]:
        """Encodes a list of prompts using the pipeline.

        Note this requires the T5 encoder has been created.
        You should free it after encoding your prompts to save memory.

        Args:
            prompts: a list of prompts to encode.
            negative_prompts: not used; included for compatibility with other image generators.
            batch_size: the batch size to use for encoding the prompts. If None, the entire list will be encoded at once.

        Returns:
            A PixArtPromptEmbedding dictionary with each value as a tensor.
        """
        if negative_prompts is not None:
            print(
                "WARNING: negative prompts are not used in PixArt; ignoring."
            )

        if self.encoder_pipeline is None:
            print(
                "WARNING: Encoder pipeline not initialized. Initializing now."
            )
            self.create_encoder_pipeline()
            assert self.encoder_pipeline is not None

        if batch_size is None:
            batch_size = len(prompts)

        # lists of:
        #   prompt_embeds,
        #   prompt_attention_mask,
        #   negative_prompt_embeds,
        #   negative_prompt_attention_mask
        all_embeds = [[], [], [], []]
        with torch.no_grad():
            for i in range(0, len(prompts), batch_size):
                self.random_generator.manual_seed(
                    self.start_seed + i * self.seed_step
                )
                print(
                    f"Encoding batch {(i // batch_size) + 1} of {(len(prompts) // batch_size)}"
                    f" with seed {self.random_generator.initial_seed()}."
                )
                batch_prompts = prompts[i : i + batch_size]
                # Note: we don't use CFG here because it will be called internally during image generation
                batch_embeds = list(
                    self.encoder_pipeline.encode_prompt(
                        batch_prompts,
                        negative_prompt="",
                        do_classifier_free_guidance=False,
                        generator=self.random_generator,
                    )
                )
                for j in range(len(all_embeds)):
                    all_embeds[j].append(
                        batch_embeds[j].to("cpu")
                        if batch_embeds[j] is not None
                        else None
                    )

                # free up some memory
                del batch_embeds
                gc.collect()
                torch.cuda.empty_cache()

        for i in range(len(all_embeds)):
            if None in all_embeds[i]:
                all_embeds[i] = None
            else:
                all_embeds[i] = torch.cat(all_embeds[i])

        # negative prompt embeddings are not generated when using CFG
        # but we need to generate them for "" for PixArt manually
        if all_embeds[2] is None or all_embeds[3] is None:
            print(
                "Missing negative prompt embeddings; using embeddings for ''."
            )
            all_embeds[2] = self.negative_prompt_embeds.repeat(
                all_embeds[0].shape[0], 1, 1
            ).to("cpu")
            all_embeds[3] = self.negative_prompt_attention_mask.repeat(
                all_embeds[1].shape[0], 1
            ).to("cpu")

        embedding_by_prompt = []

        for idx, embed_tuple in enumerate(zip(*all_embeds)):
            (
                prompt_embed,
                prompt_attention_mask,
                negative_embed,
                negative_prompt_attention_mask,
            ) = embed_tuple
            if (
                prompt_embed is None
                or prompt_attention_mask is None
                or negative_embed is None
                or negative_prompt_attention_mask is None
            ):
                raise ValueError(f"Prompt embedding is None for prompt {idx}.")

            embedding_by_prompt.append(
                {
                    "prompt_embeds": prompt_embed.clone(),
                    "prompt_attention_mask": prompt_attention_mask.clone(),
                    "negative_prompt_embeds": negative_embed.clone(),
                    "negative_prompt_attention_mask": negative_prompt_attention_mask.clone(),
                }
            )

        return embedding_by_prompt

    @torch.inference_mode()
    def encode_and_save_prompts(
        self,
        name_to_prompt: dict[str, dict[str, str]] | dict[str, str],
        output_dir: Path,
        free_after: bool = False,
        batch_size: int | None = None,
    ):
        if not name_to_prompt:
            print("No prompts found.")
            return

        # convert to just prompt strings, since we don't use negatives
        if name_to_prompt and isinstance(
            next(iter(name_to_prompt.values())), dict
        ):
            name_to_prompt = {
                name: prompt["prompt"]
                for name, prompt in name_to_prompt.items()
            }

        if self.encoder_pipeline is None:
            print("Creating encoder pipeline.")
            self.create_encoder_pipeline()

        print("Encoding prompts.")
        embedded_prompts = self.encode_prompts(
            list(name_to_prompt.values()), batch_size=batch_size
        )

        if free_after:
            print("Freeing encoder pipeline.")
            self.free_encoder_pipeline()

        for name, prompt in zip(name_to_prompt.keys(), embedded_prompts):
            output_fname = Path(name)
            if output_fname.suffix != ".pt":
                output_fname = Path(f"{output_fname}.pt")
            output_path = output_dir / output_fname
            print(f"Saving prompt to '{output_path}'.")
            torch.save(prompt, output_path)

    @torch.inference_mode()
    def generate_images(
        self,
        prompt_embeds: PixArtPromptEmbedding,
        images_per_prompt: int = 1,
        height: int | None = None,
        width: int | None = None,
        **kwargs,
    ) -> list[list[Image]]:
        """Generates all images from the provided prompts using the pipeline in parallel.

        Args:
            **kwargs:
            prompt_embeds: a Tensor of shape (4, -1), where the first four slices are prompt_embeds,
                prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask.
            images_per_prompt: the number of images to generate per prompt.
                Note these are generated sequentially, always, to avoid a bug producing random noise images.

        Note:
            The number of inference steps to run is determined by the DiT scheduler.

        Returns:
            A list (of len prompts/batch size) of lists (of len number of images per prompt) of PIL Image objects,
                where each inner list corresponds to one prompt.
        """
        if self.diffusion_pipeline is None:
            raise ValueError("PixArt pipeline not initialized.")

        height = height or self.height
        width = width or self.width

        all_images = []
        with torch.no_grad():
            for i in range(images_per_prompt):

                self.random_generator.manual_seed(
                    self.start_seed + i * self.seed_step
                )

                print(
                    f"Generating image {i} of {images_per_prompt - 1} per prompt"
                    f" with seed {self.random_generator.initial_seed()}."
                )

                batch_images: list[Image] = self.diffusion_pipeline(
                    prompt=None,  # type: ignore
                    negative_prompt=None,  # type: ignore
                    prompt_embeds=prompt_embeds["prompt_embeds"].to(
                        self.device
                    ),
                    prompt_attention_mask=prompt_embeds[
                        "prompt_attention_mask"
                    ].to(self.device),
                    negative_prompt_embeds=prompt_embeds[
                        "negative_prompt_embeds"
                    ].to(self.device),
                    negative_prompt_attention_mask=prompt_embeds[
                        "negative_prompt_attention_mask"
                    ].to(self.device),
                    num_images_per_prompt=1,
                    num_inference_steps=self.num_inference_steps,
                    generator=self.random_generator,
                    return_dict=False,
                    guidance_scale=4.5,
                    height=height,
                    width=width,
                    # very important to track timesteps for custom caching schedule
                    callback=self._call_callbacks_wrapper,
                    callback_steps=1,
                )[0]
                all_images.append(batch_images)

        # swap dimensions to have a list of images per prompt
        reshaped = [[] for _ in range(len(all_images[0]))]

        for kth_gen in all_images:
            for i, image in enumerate(kth_gen):
                reshaped[i].append(image)

        return reshaped

    @torch.inference_mode()
    def generate_images_timed(
        self,
        prompt_embeds: PixArtPromptEmbedding,
        **kwargs,
    ) -> float:
        if self.diffusion_pipeline is None:
            raise ValueError("PixArt pipeline not initialized.")

        self.random_generator.manual_seed(self.start_seed)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        with torch.no_grad():
            print(
                f"Timing image generation with seed {self.random_generator.initial_seed()}."
            )

            start.record()
            _ = self.diffusion_pipeline(
                prompt=None,  # type: ignore
                negative_prompt=None,  # type: ignore
                prompt_embeds=prompt_embeds["prompt_embeds"].to(self.device),
                prompt_attention_mask=prompt_embeds[
                    "prompt_attention_mask"
                ].to(self.device),
                negative_prompt_embeds=prompt_embeds[
                    "negative_prompt_embeds"
                ].to(self.device),
                negative_prompt_attention_mask=prompt_embeds[
                    "negative_prompt_attention_mask"
                ].to(self.device),
                num_images_per_prompt=1,
                num_inference_steps=self.num_inference_steps,
                generator=self.random_generator,
                return_dict=False,
                guidance_scale=4.5,
                # very important to track timesteps for custom caching schedule
                callback=self._call_callbacks_wrapper,
                callback_steps=1,
            )
            end.record()
            torch.cuda.synchronize()
            elapsed_time = start.elapsed_time(end)
            elapsed_time /= prompt_embeds["prompt_embeds"].shape[0]

        return elapsed_time
