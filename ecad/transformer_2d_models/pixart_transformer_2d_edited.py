import os
from typing import Any, Dict, Optional

import torch
from torch import nn
from ecad.transformer_blocks.cached_transformer_block import (
    CachedTransformerBlock,
)
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.transformers.pixart_transformer_2d import (
    PixArtTransformer2DModel,
)

from ecad.schedulers.cache_scheduler.pixart_cache_schedule import (
    PixArtCacheSchedule,
)
from ecad.schedulers.dit_scheduler.dit_scheduler import DiTScheduler
from ecad.utils import PixArtForwardArgs


class PixArtTransformer2DEdited(PixArtTransformer2DModel):

    def __init__(
        self,
        num_attention_heads: int = 16,
        attention_head_dim: int = 72,
        in_channels: int = 4,
        out_channels: Optional[int] = 8,
        num_layers: int = 28,
        dropout: float = 0.0,
        norm_num_groups: int = 32,
        cross_attention_dim: Optional[int] = 1152,
        attention_bias: bool = True,
        sample_size: int = 128,
        patch_size: int = 2,
        activation_fn: str = "gelu-approximate",
        num_embeds_ada_norm: Optional[int] = 1000,
        upcast_attention: bool = False,
        norm_type: str = "ada_norm_single",
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-6,
        interpolation_scale: Optional[int] = None,
        use_additional_conditions: Optional[bool] = None,
        caption_channels: Optional[int] = None,
        attention_type: Optional[str] = "default",
        # above are passed onto the parent class
        dit_scheduler: DiTScheduler | None = None,
        cache_schedule: PixArtCacheSchedule | None = None,
    ):
        super().__init__(
            num_attention_heads,
            attention_head_dim,
            in_channels,
            out_channels,
            num_layers,
            dropout,
            norm_num_groups,
            cross_attention_dim,
            attention_bias,
            sample_size,
            patch_size,
            activation_fn,
            num_embeds_ada_norm,
            upcast_attention,
            norm_type,
            norm_elementwise_affine,
            norm_eps,
            interpolation_scale,
            use_additional_conditions,
            caption_channels,
            attention_type,
        )

        # we cannot make this a required parameter to properly override from_pretrained in parent class
        self.dit_scheduler: DiTScheduler = dit_scheduler  # type: ignore
        self.cache_schedule: PixArtCacheSchedule | None = cache_schedule

        # need to save norm_type for CachedTransformerBlock
        self.norm_type = norm_type

    def _post_init(
        self,
        dit_scheduler: DiTScheduler | None,
        cache_schedule: PixArtCacheSchedule | None,
    ):
        if dit_scheduler is None:
            raise ValueError("A DiTScheduler object must be provided.")

        # swap in cached transformer blocks if cache_schedule is provided
        self.cache_schedule = cache_schedule
        if self.cache_schedule is not None:
            print("Using cached transformer blocks.")
            self.init_cached_transformer_blocks()
        else:
            print("Using basic transformer blocks.")

        # ensure half precision for transformer blocks
        self.transformer_blocks.half()

        # rebuilds the DiT schedule graphs with new transformer blocks
        dit_scheduler.update_transformer_blocks(self.transformer_blocks)
        self.dit_scheduler = dit_scheduler

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[str | os.PathLike],
        dit_scheduler: DiTScheduler | None = None,
        cache_schedule: PixArtCacheSchedule | None = None,
        **kwargs,
    ) -> "PixArtTransformer2DEdited":
        model: PixArtTransformer2DEdited = super().from_pretrained(
            pretrained_model_name_or_path, **kwargs
        )  # type: ignore

        model._post_init(dit_scheduler, cache_schedule)
        return model

    def init_cached_transformer_blocks(self) -> None:
        if self.cache_schedule is None:
            raise ValueError(
                "A CacheSchedule object must be provided if initializing caching transformer blocks."
            )

        self.config: Any  # type: ignore to make type checker happy

        # replace the current transformer blocks with CachedTransformerBlocks
        new_transformer_blocks = []
        for block_num, basic_block in enumerate(self.transformer_blocks):
            # Create a CachedTransformerBlock with the same parameters
            cached_block = CachedTransformerBlock(
                str(block_num),
                self.cache_schedule,
                self.inner_dim,
                self.config.num_attention_heads,
                self.config.attention_head_dim,
                dropout=self.config.dropout,
                cross_attention_dim=self.config.cross_attention_dim,
                activation_fn=self.config.activation_fn,
                num_embeds_ada_norm=self.config.num_embeds_ada_norm,
                attention_bias=self.config.attention_bias,
                upcast_attention=self.config.upcast_attention,
                norm_type=self.norm_type,
                norm_elementwise_affine=self.config.norm_elementwise_affine,
                norm_eps=self.config.norm_eps,
                attention_type=self.config.attention_type,
            )

            # transfer weights from the original block to the cached block
            cached_block.load_state_dict(basic_block.state_dict(), strict=True)
            new_transformer_blocks.append(cached_block)

        self.transformer_blocks = nn.ModuleList(new_transformer_blocks)

    def reset_cache(self) -> None:

        for block in self.transformer_blocks:
            block.reset_cache()

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        added_cond_kwargs: Dict[str, torch.Tensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> tuple[torch.Tensor] | Transformer2DModelOutput:
        """
        The [`PixArtTransformer2DModel`] forward method, overriden to follow a custom caching schedule.

        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch size, channel, height, width)`):
                Input `hidden_states`.
            encoder_hidden_states (`torch.FloatTensor` of shape `(batch size, sequence len, embed dims)`, *optional*):
                Conditional embeddings for cross attention layer. If not given, cross-attention defaults to
                self-attention.
            timestep (`torch.LongTensor`, *optional*):
                Used to indicate denoising step. Optional timestep to be applied as an embedding in `AdaLayerNorm`.
            added_cond_kwargs: (`Dict[str, Any]`, *optional*): Additional conditions to be used as inputs.
            cross_attention_kwargs ( `Dict[str, Any]`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            attention_mask ( `torch.Tensor`, *optional*):
                An attention mask of shape `(batch, key_tokens)` is applied to `encoder_hidden_states`. If `1` the mask
                is kept, otherwise if `0` it is discarded. Mask will be converted into a bias, which adds large
                negative values to the attention scores corresponding to "discard" tokens.
            encoder_attention_mask ( `torch.Tensor`, *optional*):
                Cross-attention mask applied to `encoder_hidden_states`. Two formats supported:

                    * Mask `(batch, sequence_length)` True = keep, False = discard.
                    * Bias `(batch, 1, sequence_length)` 0 = keep, -10000 = discard.

                If `ndim == 2`: will be interpreted as a mask, then converted into a bias consistent with the format
                above. This bias will be added to the cross-attention scores.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.unets.unet_2d_condition.UNet2DConditionOutput`] instead of a plain
                tuple.

        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """

        if self.use_additional_conditions and added_cond_kwargs is None:
            raise ValueError(
                "`added_cond_kwargs` cannot be None when using additional conditions for `adaln_single`."
            )

        attention_mask, encoder_attention_mask = self._create_attention_mask(
            hidden_states, attention_mask, encoder_attention_mask
        )

        # 1. Input
        (
            height,
            width,
            hidden_states,
            encoder_hidden_states,
            timestep,
            embedded_timestep,
        ) = self._process_input(
            hidden_states, encoder_hidden_states, timestep, added_cond_kwargs
        )

        # 2. Blocks
        forward_args = PixArtForwardArgs(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            timestep=timestep,
            cross_attention_kwargs=cross_attention_kwargs,
        )
        if self.training and self.gradient_checkpointing:
            for block in self.transformer_blocks:
                hidden_states = self._train_block_step(
                    block,
                    **forward_args.to_dict(),
                )
        else:
            # step forward using custom caching schedule
            hidden_states = self.dit_scheduler.forward(
                forward_args,
            )

        # 3. Output
        return self._create_output(
            hidden_states, embedded_timestep, height, width, return_dict
        )

    def _create_attention_mask(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        # ensure attention_mask is a bias, and give it a singleton query_tokens dimension.
        #   we may have done this conversion already, e.g. if we came here via UNet2DConditionModel#forward.
        #   we can tell by counting dims; if ndim == 2: it's a mask rather than a bias.
        # expects mask of shape:
        #   [batch, key_tokens]
        # adds singleton query_tokens dimension:
        #   [batch,                    1, key_tokens]
        # this helps to broadcast it as a bias over attention scores, which will be in one of the following shapes:
        #   [batch,  heads, query_tokens, key_tokens] (e.g. torch sdp attn)
        #   [batch * heads, query_tokens, key_tokens] (e.g. xformers or classic attn)
        if attention_mask is not None and attention_mask.ndim == 2:
            # assume that mask is expressed as:
            #   (1 = keep,      0 = discard)
            # convert mask into a bias that can be added to attention scores:
            #       (keep = +0,     discard = -10000.0)
            attention_mask = (
                1 - attention_mask.to(hidden_states.dtype)
            ) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # convert encoder_attention_mask to a bias the same way we do for attention_mask
        if (
            encoder_attention_mask is not None
            and encoder_attention_mask.ndim == 2
        ):
            encoder_attention_mask = (
                1 - encoder_attention_mask.to(hidden_states.dtype)
            ) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        return attention_mask, encoder_attention_mask

    def _process_input(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        added_cond_kwargs: Dict[str, torch.Tensor] = None,
    ):

        batch_size = hidden_states.shape[0]
        height, width = (
            hidden_states.shape[-2] // self.config.patch_size,
            hidden_states.shape[-1] // self.config.patch_size,
        )
        hidden_states = self.pos_embed(hidden_states)

        timestep, embedded_timestep = self.adaln_single(
            timestep,
            added_cond_kwargs,
            batch_size=batch_size,
            hidden_dtype=hidden_states.dtype,
        )

        if self.caption_projection is not None:
            encoder_hidden_states = self.caption_projection(
                encoder_hidden_states
            )
            encoder_hidden_states = encoder_hidden_states.view(
                batch_size, -1, hidden_states.shape[-1]
            )

        return (
            height,
            width,
            hidden_states,
            encoder_hidden_states,
            timestep,
            embedded_timestep,
        )

    def _create_output(
        self,
        hidden_states: torch.Tensor,
        embedded_timestep: torch.Tensor,
        height: int,
        width: int,
        return_dict: bool,
    ) -> tuple[torch.Tensor] | Transformer2DModelOutput:
        shift, scale = (
            self.scale_shift_table[None]
            + embedded_timestep[:, None].to(self.scale_shift_table.device)
        ).chunk(2, dim=1)
        hidden_states = self.norm_out(hidden_states)
        # Modulation
        hidden_states = hidden_states * (
            1 + scale.to(hidden_states.device)
        ) + shift.to(hidden_states.device)
        hidden_states = self.proj_out(hidden_states)
        hidden_states = hidden_states.squeeze(1)

        # unpatchify
        hidden_states = hidden_states.reshape(
            shape=(
                -1,
                height,
                width,
                self.config.patch_size,
                self.config.patch_size,
                self.out_channels,
            )
        )
        hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
        output = hidden_states.reshape(
            shape=(
                -1,
                self.out_channels,
                height * self.config.patch_size,
                width * self.config.patch_size,
            )
        )

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)

    def _train_block_step(
        self,
        block: torch.nn.Module,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
    ):
        def create_custom_forward(module, return_dict=None):
            def custom_forward(*inputs):
                if return_dict is not None:
                    return module(*inputs, return_dict=return_dict)
                else:
                    return module(*inputs)

            return custom_forward

        # we are pinned on torch version >= 1.11
        ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False}
        hidden_states = torch.utils.checkpoint.checkpoint(
            create_custom_forward(block),
            hidden_states,
            attention_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            timestep,
            cross_attention_kwargs,
            None,
            **ckpt_kwargs,
        )
        return hidden_states
