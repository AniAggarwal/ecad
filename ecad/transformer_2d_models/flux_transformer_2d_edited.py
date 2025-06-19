import gc
import os
from typing import Any, Optional
from diffusers.models.transformers.transformer_flux import (
    FluxTransformer2DModel,
)
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils import (
    is_torch_version,
    logging,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.utils.constants import USE_PEFT_BACKEND
import torch
from tqdm import tqdm

from ecad.schedulers.cache_scheduler.flux_cache_schedule import (
    FluxCacheSchedule,
)
from ecad.schedulers.dit_scheduler.dit_scheduler import DiTScheduler
from ecad.transformer_blocks.cached_flux_transformer_block import (
    CachedFluxSingleTransformerBlock,
    CachedFluxTransformerBlock,
)

from torch import nn
import torch.utils.checkpoint

from ecad.utils import FluxForwardArgs

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class FluxTransformer2DEdited(FluxTransformer2DModel):
    def __init__(
        self,
        patch_size: int = 1,
        in_channels: int = 64,
        num_layers: int = 19,
        num_single_layers: int = 38,
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        joint_attention_dim: int = 4096,
        pooled_projection_dim: int = 768,
        guidance_embeds: bool = False,
        axes_dims_rope: list[int] = [16, 56, 56],
        # above are passed onto the parent class
        dit_scheduler: DiTScheduler | None = None,
        cache_schedule: FluxCacheSchedule | None = None,
    ):

        super().__init__(
            patch_size,
            in_channels,
            num_layers,
            num_single_layers,
            attention_head_dim,
            num_attention_heads,
            joint_attention_dim,
            pooled_projection_dim,
            guidance_embeds,
            axes_dims_rope,
        )

        # we cannot make this a required parameter to properly override from_pretrained in parent class
        self.dit_scheduler: DiTScheduler = dit_scheduler  # type: ignore
        self.cache_schedule: FluxCacheSchedule | None = cache_schedule

    @torch.no_grad()
    def _post_init(
        self,
        dit_scheduler: DiTScheduler | None,
        cache_schedule: FluxCacheSchedule | None,
        skip_transformer_block_init: bool = False,
    ):
        if dit_scheduler is None:
            raise ValueError("A DiTScheduler object must be provided.")

        if skip_transformer_block_init:
            print(
                "WARNING: Skipping transformer block initialization. Generated images will be random noise."
            )
        # swap in cached transformer blocks if cache_schedule is provided
        self.cache_schedule = cache_schedule
        if self.cache_schedule is not None:
            print("Using cached transformer blocks.")
            self.init_cached_transformer_blocks(skip_transformer_block_init)
        else:
            print("Using basic transformer blocks.")

        # ensure bfloat16 for transformer blocks
        self.transformer_blocks = self.transformer_blocks.to(
            dtype=torch.bfloat16
        )
        self.single_transformer_blocks = self.single_transformer_blocks.to(
            dtype=torch.bfloat16
        )

        # rebuilds the DiT schedule graphs with new transformer blocks
        dit_scheduler.update_transformer_blocks(
            transformer_blocks=self.transformer_blocks,
            single_transformer_blocks=self.single_transformer_blocks,
        )
        self.dit_scheduler = dit_scheduler
        print("FluxTransformer2DEdited model initialized.")

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[str | os.PathLike],
        dit_scheduler: DiTScheduler | None = None,
        cache_schedule: FluxCacheSchedule | None = None,
        skip_transformer_block_init: bool = False,
        **kwargs,
    ) -> "FluxTransformer2DEdited":
        model: FluxTransformer2DEdited = super().from_pretrained(
            pretrained_model_name_or_path, **kwargs
        )  # type: ignore

        model._post_init(
            dit_scheduler, cache_schedule, skip_transformer_block_init
        )
        return model

    @torch.no_grad()
    def init_cached_transformer_blocks(
        self, skip_transformer_block_init: bool = False
    ) -> None:
        if self.cache_schedule is None:
            raise ValueError(
                "A CacheSchedule object must be provided if initializing caching transformer blocks."
            )

        self.config: Any  # type: ignore to make type checker happy - set by huggingface

        # TODO: can we speed up this method? it is very slow
        # perhaps it is bc the original transformers are on GPU
        # and their weights are beign offloaded, then copied?

        # replace the current transformer blocks with cached ones, in place to reduce memory usage
        print("Initializing cached transformer blocks.")
        self.transformer_blocks = self.transformer_blocks.to("cpu")
        for block_num in tqdm(range(len(self.transformer_blocks))):
            # Create a cached block with the same parameters
            self.transformer_blocks[block_num] = CachedFluxTransformerBlock(
                block_num=str(block_num),
                cache_schedule=self.cache_schedule,
                dim=self.inner_dim,
                num_attention_heads=self.config.num_attention_heads,
                attention_head_dim=self.config.attention_head_dim,
                state_dict=(
                    self.transformer_blocks[block_num].state_dict()
                    if not skip_transformer_block_init
                    else None
                ),
            )
            gc.collect()
            torch.cuda.empty_cache()

        print("Initializing single cached transformer blocks.")
        for block_num in tqdm(range(len(self.single_transformer_blocks))):
            # Create a cached block with the same parameters
            self.single_transformer_blocks[block_num] = (
                CachedFluxSingleTransformerBlock(
                    block_num=f"single_{block_num}",
                    cache_schedule=self.cache_schedule,
                    dim=self.inner_dim,
                    num_attention_heads=self.config.num_attention_heads,
                    attention_head_dim=self.config.attention_head_dim,
                    state_dict=(
                        self.single_transformer_blocks[block_num].state_dict()
                        if not skip_transformer_block_init
                        else None
                    ),
                )
            )
            gc.collect()
            torch.cuda.empty_cache()

        print("Done initializing cached transformer blocks.")

    def reset_cache(self) -> None:
        for block in self.transformer_blocks:
            block.reset_cache()

        for single_block in self.single_transformer_blocks:
            single_block.reset_cache()


    def _dit_scheduler_forward_short_circuit(
        self, forward_args: FluxForwardArgs
    ) -> torch.Tensor:
        hidden_states = forward_args.hidden_states
        encoder_hidden_states = forward_args.encoder_hidden_states
        temb = forward_args.temb
        image_rotary_emb = forward_args.image_rotary_emb

        for index_block, block in enumerate(self.transformer_blocks):
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
            )

        hidden_states = torch.cat(
            [encoder_hidden_states, hidden_states], dim=1
        )

        for index_block, block in enumerate(self.single_transformer_blocks):
            hidden_states = block(
                hidden_states=hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
            )

        return hidden_states

    def forward(  # type: ignore since base forward' type annotation is missing tuple
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        pooled_projections: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_ids: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        guidance: torch.Tensor = None,
        joint_attention_kwargs: Optional[dict[str, Any]] = None,
        return_dict: bool = True,
    ) -> tuple[torch.FloatTensor] | Transformer2DModelOutput:
        """
        The [`FluxTransformer2DModel`] forward method.

        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch size, channel, height, width)`):
                Input `hidden_states`.
            encoder_hidden_states (`torch.FloatTensor` of shape `(batch size, sequence_len, embed_dims)`):
                Conditional embeddings (embeddings computed from the input conditions such as prompts) to use.
            pooled_projections (`torch.FloatTensor` of shape `(batch_size, projection_dim)`): Embeddings projected
                from the embeddings of input conditions.
            timestep ( `torch.LongTensor`):
                Used to indicate denoising step.
            block_controlnet_hidden_states: (`list` of `torch.Tensor`):
                A list of tensors that if specified are added to the residuals of transformer blocks.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.transformer_2d.Transformer2DModelOutput`] instead of a plain
                tuple.

        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """
        if joint_attention_kwargs is not None:
            joint_attention_kwargs = joint_attention_kwargs.copy()
            lora_scale = joint_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if (
                joint_attention_kwargs is not None
                and joint_attention_kwargs.get("scale", None) is not None
            ):
                logger.warning(
                    "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
                )
        hidden_states = self.x_embedder(hidden_states)

        timestep = timestep.to(hidden_states.dtype) * 1000
        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000
        else:
            guidance = None
        temb = (
            self.time_text_embed(timestep, pooled_projections)
            if guidance is None
            else self.time_text_embed(timestep, guidance, pooled_projections)
        )
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        ids = torch.cat((txt_ids, img_ids), dim=1)
        image_rotary_emb = self.pos_embed(ids)

        # NOTE: the main modification is this block of code
        forward_args = FluxForwardArgs(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            temb=temb,
            image_rotary_emb=image_rotary_emb,
        )
        if self.training and self.gradient_checkpointing:
            hidden_states, encoder_hidden_states = self._train_blocks(
                **forward_args.to_dict()
            )
        else:
            # hidden_states = self.dit_scheduler.forward(
            #     forward_args,
            # )
            hidden_states = self._dit_scheduler_forward_short_circuit(
                forward_args,
            )
        # END OF MODIFICATION

        # NOTE: if there is a shape issue here, then we will need to update code to
        # return both hidden_states and encoder_hidden_states from the transformer blocks
        hidden_states = hidden_states[:, encoder_hidden_states.shape[1] :, ...]

        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)

    def _train_blocks(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: torch.Tensor,
    ) -> torch.Tensor:
        ckpt_kwargs: dict[str, Any]

        for index_block, block in enumerate(self.transformer_blocks):

            def create_custom_forward(module, return_dict=None):
                def custom_forward(*inputs):
                    if return_dict is not None:
                        return module(*inputs, return_dict=return_dict)
                    else:
                        return module(*inputs)

                return custom_forward

            ckpt_kwargs = (
                {"use_reentrant": False}
                if is_torch_version(">=", "1.11.0")
                else {}
            )
            encoder_hidden_states, hidden_states = (
                torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    image_rotary_emb,
                    **ckpt_kwargs,
                )
            )

        hidden_states = torch.cat(
            [encoder_hidden_states, hidden_states], dim=1
        )

        for index_block, block in enumerate(self.single_transformer_blocks):

            def create_custom_forward(module, return_dict=None):
                def custom_forward(*inputs):
                    if return_dict is not None:
                        return module(*inputs, return_dict=return_dict)
                    else:
                        return module(*inputs)

                return custom_forward

            ckpt_kwargs = (
                {"use_reentrant": False}
                if is_torch_version(">=", "1.11.0")
                else {}
            )
            hidden_states = torch.utils.checkpoint.checkpoint(
                create_custom_forward(block),
                hidden_states,
                temb,
                image_rotary_emb,
                **ckpt_kwargs,
            )

        return hidden_states
