from abc import ABC, abstractmethod
import gc
from pathlib import Path
from typing import Any, Callable, Sequence
import torch

from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from PIL.Image import Image
from torch.utils.data import DataLoader

from ecad.schedulers.cache_scheduler.cache_schedule import (
    CacheSchedule,
)
from ecad.schedulers.dit_scheduler.dit_schedule import DiTSchedule
from ecad.schedulers.dit_scheduler.dit_scheduler import DiTScheduler
from ecad.pipelines.load_pipeline import pipeline_from_pretrained
from ecad.dataset_utils.prompt_embedding_dataset import (
    PromptEmbeddingDataset,
)
from ecad.types import (
    ImageGeneratorConfig,
    PromptEmbedding,
    PromptEmbeddingType,
    PromptEmbeddingType,
    PromptEmbeddingType,
)


class ImageGenerator(ABC):
    """
    Base class for image generators. Provides common functionality for different
    image generation models like PixArt and FLUX.
    """

    def __init__(
        self,
        default_transformer_weights: str = None,
        default_pipeline_weights: str = None,
        default_pipeline_name: str = None,
        schedule_path: Path | None = None,
        start_seed: int = 0,
        seed_step: int = 1,
        device: str = "cuda",
        additional_callbacks: list[Callable] | None = None,
        dit_schedule_type: type[DiTSchedule] = None,
        cache_schedule_type: type[CacheSchedule] = None,
    ):
        """
        Initialize the image generator.

        Args:
            default_transformer_weights: Name of transformer model weights if not specified in schedule file
            default_pipeline_weights: Name or pipeline model weights if not specified in schedule file
            default_pipeline_name: Name of the default pipeline to use if not specified in schedule file
            schedule_path: Path to the schedule file
            seed: Initial random seed for generation
            seed_step: Step size for incrementing seed between generations
            device: Device to run the model on ("cuda", "cpu", etc.)
            additional_callbacks: List of callback functions to be called during generation
                other than the required ones for the DiT and Cache schedules
        """
        self.device: torch.device = torch.device(device)
        self.default_transformer_weights: str = default_transformer_weights
        self.default_pipeline_weights: str = default_pipeline_weights
        self.default_pipeline_name: str = default_pipeline_name
        self.schedule_path: Path | None = schedule_path
        self.start_seed: int = start_seed
        self.seed_step: int = seed_step
        self.additional_callbacks: list[Callable] = additional_callbacks or []

        self.dit_schedule_type: type[DiTSchedule] = dit_schedule_type
        self.cache_schedule_type: type[CacheSchedule] = cache_schedule_type

        self.encoder_pipeline: DiffusionPipeline | None = None
        self.diffusion_pipeline: DiffusionPipeline | None = None

        # these will be set in _load_schedule_file
        self.random_generator: torch.Generator
        self.dit_scheduler: DiTScheduler
        self.cache_schedule: CacheSchedule
        self.num_inference_steps: int
        self.pipeline_from_pretrained: Callable[..., DiffusionPipeline]
        self.transformer_weights: str
        self.pipeline_weights: str

        self._initialize_random_generator()
        self._load_schedule_file()

    def _initialize_random_generator(self):
        """Initialize the random number generator with the start seed."""
        # always use CPU generator to ensure reproducibility
        self.random_generator = torch.Generator(device="cpu")
        self.random_generator.manual_seed(self.start_seed)
        print(
            f"Using random start seed: {self.start_seed}. "
            f"Stepping seed by {self.seed_step} for each successive image per prompt."
        )

    def _load_schedule_file(self):
        """
        Load schedules for diffusion process.
        """
        dit_was_none = True
        cache_was_none = True

        if self.schedule_path is None:
            dit_schedule = None
            cache_schedule = None
        else:
            try:
                dit_schedule = self.dit_schedule_type.from_json(
                    self.schedule_path
                )
                dit_was_none = False
            except KeyError as e:
                dit_schedule = None

            try:
                cache_schedule = self.cache_schedule_type.from_json(
                    self.schedule_path
                )
                cache_was_none = False
            except KeyError as e:
                cache_schedule = None

        if dit_schedule is None:
            dit_schedule = self._default_dit_schedule()
        if cache_schedule is None:
            cache_schedule = self._default_cache_schedule()

        self.dit_scheduler = DiTScheduler(dit_schedule)
        self.cache_schedule = cache_schedule
        print(f"Using DiT schedule: {dit_schedule.name}.")
        print(f"Using cache schedule: {self.cache_schedule.name}.")

        if dit_was_none and cache_was_none:
            dit_n_steps = self.dit_scheduler.schedule.num_inference_steps
            cache_n_steps = self.cache_schedule.num_inference_steps
            if dit_n_steps != cache_n_steps:
                raise ValueError(
                    "DiT and cache schedules have different numbers of inference steps; should be impossible."
                )
            self.num_inference_steps = dit_n_steps
        if dit_was_none and not cache_was_none:
            self.num_inference_steps = self.cache_schedule.num_inference_steps
        if not dit_was_none and cache_was_none:
            self.num_inference_steps = (
                self.dit_scheduler.schedule.num_inference_steps
            )

        print(f"Using {self.num_inference_steps} inference steps.")

        self.callbacks = [
            self.dit_scheduler.per_step_callback,
            self.cache_schedule.per_step_callback,
        ]
        self.callbacks.extend(self.additional_callbacks)
        # IMPORTANT: reset MUST be LAST in the list
        self.callbacks.append(self._reset_schedules_callback)

        dit_config = dit_schedule.top_level_config
        cache_config = cache_schedule.top_level_config
        if dit_config and cache_config and dit_config != cache_config:
            raise ValueError(
                "DiT and cache schedules have different top level configs; should be impossible."
            )
        config: ImageGeneratorConfig = (
            dit_config if dit_config else cache_config
        )  # type: ignore
        self._load_config(config)

    def _load_config(self, config: ImageGeneratorConfig) -> None:
        self.config: ImageGeneratorConfig = config
        self.transformer_weights = config.get(
            "transformer_weights", self.default_transformer_weights
        )

        self.pipeline_weights = config.get(
            "pipeline_weights", self.default_pipeline_weights
        )

        self.pipeline_from_pretrained = pipeline_from_pretrained(
            config.get("pipeline", {}), self.default_pipeline_name
        )

        self._load_subclass_config_defaults(config)

    def _load_subclass_config_defaults(
        self, config: ImageGeneratorConfig
    ) -> None:
        pass

    def _reset_schedules_callback(
        self, step: int, timestep: int, **kwargs: Any
    ) -> None:
        # callback is called at the end of each inference step
        if step >= self.num_inference_steps - 1:
            self.dit_scheduler.reset_step()
            self.cache_schedule.reset_step()

            if self.diffusion_pipeline is not None:
                self.diffusion_pipeline.transformer.reset_cache()

    def _call_callbacks(self, step: int, timestep: int, **kwargs: Any) -> None:
        """
        Call all registered callbacks with the current state.

        Args:
            step: Current step in the diffusion process
            timestep: Current timestep tensor in the diffusion process
        """
        for callback in self.callbacks:
            callback(step, timestep, **kwargs)

    @abstractmethod
    def _default_dit_schedule(self) -> DiTSchedule:
        raise NotImplementedError(
            "Subclasses must implement _default_dit_schedule"
        )

    @abstractmethod
    def _default_cache_schedule(self) -> CacheSchedule:
        raise NotImplementedError(
            "Subclasses must implement _default_dit_schedule"
        )

    @abstractmethod
    def create_encoder_pipeline(self):
        """
        Create the encoder pipeline.
        Should be implemented by subclasses.
        """
        raise NotImplementedError(
            "Subclasses must implement create_encoder_pipeline"
        )

    @abstractmethod
    def create_diffusion_pipeline(self):
        """
        Create the diffusion image generator pipeline.
        Should be implemented by subclasses.
        """
        raise NotImplementedError(
            "Subclasses must implement create_generator_pipeline"
        )

    def free_encoder_pipeline(self):
        """
        Free resources used by the encoder pipeline.
        """
        if not hasattr(self, "encoder_pipeline"):
            raise AttributeError("encoder_pipeline not setup by subclass.")

        if self.encoder_pipeline is not None:
            del self.encoder_pipeline
            self.encoder_pipeline = None
            torch.cuda.empty_cache()
            gc.collect()
        else:
            print("WARNING: No encoder pipeline to free.")

    def free_diffusion_pipeline(self):
        """
        Free resources used by the diffusion pipeline.
        """

        if not hasattr(self, "diffusion_pipeline"):
            raise AttributeError("diffusion_pipeline not setup by subclass.")

        if self.diffusion_pipeline is not None:
            del self.diffusion_pipeline
            self.diffusion_pipeline = None
            torch.cuda.empty_cache()
            gc.collect()
        else:
            print("WARNING: No diffusion pipeline to free.")

    def load_and_batch_embeddings(
        self, embedding_dir: Path, batch_size: int, shuffle: bool = False
    ) -> DataLoader:
        """
        Loads and batches embeddings from the specified directory into a DataLoader.

        Args:
            embedding_dir (Path): Path to the directory containing the embedding files.
            batch_size (int): The size of each batch to be used during data loading.
            shuffle (bool): Whether to shuffle the dataset before batching. Defaults to False.

        Returns:
            DataLoader: A DataLoader instance configured with the specified dataset, batch size, and options.
        """
        dataset = PromptEmbeddingDataset(embedding_dir)
        dataloader = DataLoader(
            dataset,
            batch_size,
            shuffle,
            pin_memory=True,
            pin_memory_device=str(self.device),
        )
        return dataloader

    @abstractmethod
    def encode_prompts(
        self,
        prompts: list[str],
        negative_prompts: list[str] | None = None,
        batch_size: int | None = None,
        **kwargs,
    ) -> Sequence[PromptEmbedding]:
        """
        Encodes a list of textual prompts and optionally their corresponding negative prompts
        into a list of prompt embeddings. This method is abstract and needs to be implemented
        by subclasses to define the specific encoding logic.

        Args:
            prompts: A list of strings representing textual prompts to encode.
            negative_prompts: An optional list of strings representing textual negative prompts
                to encode. Defaults to None.
            batch_size: An optional integer specifying the batch size to use when encoding
                prompts. Defaults to None.

        Returns:
            A list of PromptEmbedding objects representing encoded prompts.
        """
        raise NotImplementedError("Subclasses must implement encode_prompts")

    @abstractmethod
    def encode_and_save_prompts(
        self,
        name_to_prompt: dict[str, dict[str, str]] | dict[str, str],
        output_dir: Path,
        free_after: bool = False,
        batch_size: int | None = None,
    ):
        """
        Encode prompts and save them to disk.

        Args:
            name_to_prompt: Dictionary mapping the filename to a dictionary containing the
                prompt and negative prompt, etc. as kwargs to encode_prompts.
                Or, a mapping from filename to prompt.
            output_dir: Directory to save the encoded prompts.
            free_after: Free resources and pipelines after encoding.
            batch_size: Batch size for encoding. If None, use all prompts in one batch.
        """
        raise NotImplementedError(
            "Subclasses must implement encode_and_save_prompts"
        )

    @abstractmethod
    def generate_images(
        self, prompt_embeds: PromptEmbeddingType, images_per_prompt, **kwargs
    ):
        """
        Generate images from prompt embeddings.

        Args:
            prompt_embeds: Encoded prompt embeddings
            **kwargs: Additional arguments for generation

        Returns:
            List of generated images
        """
        raise NotImplementedError("Subclasses must implement generate_images")

    @torch.inference_mode()
    def generate_from_saved_prompts(
        self,
        input_dir: Path,
        output_dir: Path,
        batch_size: int = 1,
        images_per_prompt: int = 1,
        free_after: bool = False,
        include_seed_in_name: bool = True,
        **kwargs: Any,
    ) -> None:

        if self.diffusion_pipeline is None:
            print("Creating diffusion pipeline.")
            self.create_diffusion_pipeline()

        print("Loading and batching embeddings.")
        embedding_loader = self.load_and_batch_embeddings(
            input_dir, batch_size, False
        )

        for i, embeds in enumerate(embedding_loader):
            print(f"Running batch {i} of {len(embedding_loader) - 1}.")

            # returns a list of len batch_size by images_per_prompt
            images = self.generate_images(
                embeds,
                images_per_prompt,
                **kwargs,
            )

            for name, rel_path, generations_for_one_prompt in zip(
                embeds["name"], embeds["relative_path"], images
            ):
                print(f"Saving images for prompt '{name}'.")

                for i, image in enumerate(generations_for_one_prompt):
                    image_seed = self.start_seed + i * self.seed_step

                    name_with_seed = (
                        f"{name}__image_seed:{image_seed:03}"
                        if include_seed_in_name
                        else name
                    )
                    output_path = output_dir / rel_path / name_with_seed
                    self.save_image(image, output_path)

            # free up some memory
            del embeds
            del images
            gc.collect()
            torch.cuda.empty_cache()

        if free_after:
            print("Freeing diffusion pipeline.")
            self.free_diffusion_pipeline()

    def save_image(self, image: Image, output_path: Path) -> None:
        """Saves the generated image to the output path.

        If the parent directory of the output path does not exist, it will be created.

        Args:
            image: a PIL Image object.
            output_path: a Path object representing the output file path, including the filename.
                If an extension is not present, .png will be added.
        """
        output_path = Path(output_path)

        if output_path.suffix != ".png":
            output_path = Path(f"{output_path}.png")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"Saving image to '{output_path}'.")
        image.save(output_path)

    @torch.inference_mode()
    def time_image_generation(
        self,
        input_dir: Path,
        batch_size: int = 1,
        num_batches: int = 1,
        free_after: bool = False,
        **kwargs,
    ) -> list[float]:
        if self.diffusion_pipeline is None:
            print("Creating diffusion pipeline.")
            self.create_diffusion_pipeline()

        all_times = []

        batches_run = 0
        while batches_run < num_batches:
            print("Loading and batching embeddings.")
            embedding_loader = self.load_and_batch_embeddings(
                input_dir, batch_size, False
            )

            for i, embeds in enumerate(embedding_loader):
                print(f"Running batch {i} of {len(embedding_loader) - 1}.")

                time = self.generate_images_timed(
                    embeds,
                    **kwargs,
                )
                all_times.append(time)
                print(f"Time for batch {i}: {time:.2f} ms.")

                # free up some memory
                del embeds
                gc.collect()
                torch.cuda.empty_cache()

                batches_run += 1
                if batches_run >= num_batches:
                    break

        if free_after:
            print("Freeing diffusion pipeline.")
            self.free_diffusion_pipeline()

        return all_times

    @abstractmethod
    def generate_images_timed(
        self,
        prompt_embeds: PromptEmbeddingType,
        **kwargs,
    ) -> float:
        pass
