import gc
from pathlib import Path
from typing import Any, Sequence

from PIL.Image import Image
from transformers import CLIPTextModel, T5EncoderModel
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline
import torch
from ecad.image_generators.image_generator import ImageGenerator
from ecad.schedulers.cache_scheduler.flux_cache_schedule import (
    FluxCacheSchedule,
)
from ecad.schedulers.dit_scheduler.flux_dit_schedule import (
    FluxDiTSchedule,
)
from ecad.transformer_2d_models.flux_transformer_2d_edited import (
    FluxTransformer2DEdited,
)
from ecad.types import FluxPromptEmbedding, ImageGeneratorConfig

from ecad.schedulers.dit_scheduler.generators.flux_schedule_generators import (
    gen_default as dit_gen_default,
)
from ecad.schedulers.cache_scheduler.generators.flux_schedule_generators import (
    gen_default as cache_gen_default,
)


class FluxImageGenerator(ImageGenerator):
    DEFAULT_WEIGHTS = "black-forest-labs/FLUX.1-dev"
    PIPELINE_WEIGHTS = "black-forest-labs/FLUX.1-dev"
    DEFAULT_HEIGHT = 256
    DEFAULT_WIDTH = 256
    DEFAULT_GUIDANCE_SCALE = 5
    DEFAULT_NUM_INFERENCE_STEPS = 20

    def __init__(
        self,
        weights_name: str | None = None,
        start_seed: int = 0,
        seed_step: int = 1,
        schedule_path: Path | None = None,
    ):
        if not torch.cuda.is_available():
            raise ValueError("CUDA is required for image generation.")

        weights_name = weights_name or self.DEFAULT_WEIGHTS

        super().__init__(
            default_transformer_weights=weights_name,
            default_pipeline_weights=self.PIPELINE_WEIGHTS,
            default_pipeline_name="flux",
            schedule_path=schedule_path,
            start_seed=start_seed,
            seed_step=seed_step,
            device="cuda",
            dit_schedule_type=FluxDiTSchedule,
            cache_schedule_type=FluxCacheSchedule,
        )

    def _load_subclass_config_defaults(
        self, config: ImageGeneratorConfig
    ) -> None:
        self.height = config.get("height", self.DEFAULT_HEIGHT)
        self.width = config.get("width", self.DEFAULT_WIDTH)
        self.guidance_scale = config.get(
            "guidance_scale", self.DEFAULT_GUIDANCE_SCALE
        )

    def _default_dit_schedule(self) -> FluxDiTSchedule:
        return next(dit_gen_default(19, 38, self.DEFAULT_NUM_INFERENCE_STEPS))

    def _default_cache_schedule(self) -> FluxCacheSchedule:
        return next(
            cache_gen_default(19, 38, self.DEFAULT_NUM_INFERENCE_STEPS)
        )

    def _call_callbacks_wrapper(
        self,
        _pipeline: DiffusionPipeline,
        step: int,
        timestep: int,
        callback_kwargs: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        # because the huggingface implementation for this is braindead,
        # we need to request the latents and prompt embeds from the pipeline,
        # then return them again >:(
        if (
            "latents" not in callback_kwargs
            or "prompt_embeds" not in callback_kwargs
        ):
            print(
                "WARNING: Callback kwargs missing latents or prompt embeds. This will likley cause a crash."
            )
        self._call_callbacks(step, timestep)

        return {
            "latents": callback_kwargs["latents"],
            "prompt_embeds": callback_kwargs["prompt_embeds"],
        }

    def create_encoder_pipeline(self):
        text_encoder = CLIPTextModel.from_pretrained(
            self.pipeline_weights,
            subfolder="text_encoder",
            torch_dtype=torch.bfloat16,
            use_safetensors=True,
        )

        text_encoder_2 = T5EncoderModel.from_pretrained(
            self.pipeline_weights,
            subfolder="text_encoder_2",
            torch_dtype=torch.bfloat16,
            use_safetensors=True,
        )

        encoder_pipeline: FluxPipeline = self.pipeline_from_pretrained(
            self.pipeline_weights,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            transformer=None,
            torch_dtype=torch.bfloat16,
            use_safetensors=True,
        ).to(self.device)
        self.encoder_pipeline = encoder_pipeline

    def create_diffusion_pipeline(
        self, skip_transformer_block_init: bool = False
    ) -> None:
        print("Creating Flux Transformer.")
        flux_transformer = FluxTransformer2DEdited.from_pretrained(
            self.transformer_weights,
            subfolder="transformer",
            torch_dtype=torch.bfloat16,
            use_safetensors=True,
            dit_scheduler=self.dit_scheduler,
            cache_schedule=self.cache_schedule,
            skip_transformer_block_init=skip_transformer_block_init,
        )

        print("Creating Flux pipeline.")
        pipeline: FluxPipeline = self.pipeline_from_pretrained(
            self.pipeline_weights,
            transformer=flux_transformer,
            text_encoder=None,
            text_encoder_2=None,
            torch_dtype=torch.bfloat16,
            use_safetensors=True,
        ).to(self.device)

        self.diffusion_pipeline = pipeline

    @torch.inference_mode()
    def encode_prompts(
        self,
        prompts: list[str],
        negative_prompts: list[str] | None = None,
        batch_size: int | None = None,
        include_text_ids: bool = False,
        **kwargs,
    ) -> Sequence[FluxPromptEmbedding]:
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

        # lists for: prompt_embeds, pooled_prompt_embeds, text_ids
        all_embeds = [[], [], []]
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

                batch_embeds = list(
                    self.encoder_pipeline.encode_prompt(
                        prompt=batch_prompts,
                        prompt_2=None,
                        device=self.device,
                        num_images_per_prompt=1,
                        **kwargs,
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

        embedding_by_prompt = []

        for idx, embed_tuple in enumerate(zip(*all_embeds)):
            (prompt_embed, pooled_prompt_embed, text_id) = embed_tuple
            if (
                prompt_embed is None
                or pooled_prompt_embed is None
                or text_id is None
            ):
                raise ValueError(
                    f"Part of prompt embedding is None for prompt {idx}."
                )

            flux_embed_dict = {
                "prompt_embeds": prompt_embed.clone(),
                "pooled_prompt_embeds": pooled_prompt_embed.clone(),
            }
            flux_embed_dict["text_ids"] = (
                text_id.clone() if include_text_ids else None
            )
            embedding_by_prompt.append(flux_embed_dict)

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

        # convert to just prompt strings, since we don't use negatives or a prompt 2
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
        prompt_embeds: FluxPromptEmbedding,
        images_per_prompt: int = 1,
        height: int | None = None,
        width: int | None = None,
        guidance_scale: float | None = None,
        **kwargs,
    ) -> list[list[Image]]:
        """Generates all images from the provided prompts using the pipeline in parallel.

        Args:
            **kwargs:
            prompt_embeds: a Tensor of shape (2 or 3, -1), where the first 2 (and optionally 3) slices are:
                prompt_embeds, pooled_prompt_embeds, and optionally text_ids.
            images_per_prompt: the number of images to generate per prompt.
                Note these are generated sequentially, always, to avoid a bug producing random noise images.

        Note:
            The number of inference steps to run is determined by the DiT scheduler.

        Returns:
            A list (of len prompts/batch size) of lists (of len number of images per prompt) of PIL Image objects,
                where each inner list corresponds to one prompt.
        """
        if self.diffusion_pipeline is None:
            print("Diffusion pipeline not initialized. Initializing now.")
            self.create_diffusion_pipeline()
            assert self.diffusion_pipeline is not None

        height = height or self.height
        width = width or self.width
        guidance_scale = guidance_scale or self.guidance_scale

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
                    prompt_2=None,  # type: ignore
                    prompt_embeds=prompt_embeds["prompt_embeds"].to(
                        self.device
                    ),
                    pooled_prompt_embeds=prompt_embeds[
                        "pooled_prompt_embeds"
                    ].to(self.device),
                    num_images_per_prompt=1,
                    num_inference_steps=self.num_inference_steps,
                    generator=self.random_generator,
                    return_dict=False,
                    height=height,
                    width=width,
                    guidance_scale=guidance_scale,
                    # very important to track timesteps for custom caching schedule
                    callback_on_step_end=self._call_callbacks_wrapper,
                    callback_on_step_end_tensor_inputs=[
                        "latents",
                        "prompt_embeds",
                    ],
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
        prompt_embeds: FluxPromptEmbedding,
        height: int | None = None,
        width: int | None = None,
        guidance_scale: float | None = None,
        **kwargs,
    ) -> float:
        if self.diffusion_pipeline is None:
            print("Diffusion pipeline not initialized. Initializing now.")
            self.create_diffusion_pipeline()
            assert self.diffusion_pipeline is not None

        height = height or self.height
        width = width or self.width
        guidance_scale = guidance_scale or self.guidance_scale

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
                prompt_2=None,  # type: ignore
                prompt_embeds=prompt_embeds["prompt_embeds"].to(self.device),
                pooled_prompt_embeds=prompt_embeds["pooled_prompt_embeds"].to(
                    self.device
                ),
                num_images_per_prompt=1,
                num_inference_steps=self.num_inference_steps,
                generator=self.random_generator,
                return_dict=False,
                height=height,
                width=width,
                guidance_scale=guidance_scale,
                # very important to track timesteps for custom caching schedule
                callback_on_step_end=self._call_callbacks_wrapper,
                callback_on_step_end_tensor_inputs=[
                    "latents",
                    "prompt_embeds",
                ],
            )
            end.record()
            torch.cuda.synchronize()
            elapsed_time = start.elapsed_time(end)
            elapsed_time /= prompt_embeds["prompt_embeds"].shape[0]

        return elapsed_time
