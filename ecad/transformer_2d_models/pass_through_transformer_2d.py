from typing import Any, Dict, Optional

import torch
from ecad.transformer_blocks.cached_transformer_block import CachedTransformerBlock
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.transformers.pixart_transformer_2d import (
    PixArtTransformer2DModel,
)


class PassThroughTransformer2D(PixArtTransformer2DModel):

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
    ):
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

        batch_size, channels, height, width = hidden_states.shape
        target_shape = (
            batch_size,
            self.out_channels,
            height * self.config.patch_size,
            width * self.config.patch_size,
        )

        # output = hidden_states.reshape(
        #     shape=(
        #         -1,
        #         self.out_channels,
        #         height * self.config.patch_size,
        #         width * self.config.patch_size,
        #     )
        # )
        # output = self.match_tensor_size(hidden_states, target_shape)

        output = torch.zeros(
            *target_shape,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )

        if return_dict:
            return Transformer2DModelOutput(sample=output)
        else:
            return (output,)

    # @staticmethod
    # def reshape_tensor(x, out_channels, patch_size):
    #     # x has shape: (batch, c, h, w)
    #     batch, c, h, w = x.shape
    #
    #     # Adjust channel dimension if needed
    #     if c != out_channels:
    #         if out_channels % c == 0:
    #             # Duplicate channels along the channel dimension
    #             repeat_factor = out_channels // c
    #             x = x.repeat(1, repeat_factor, 1, 1)
    #         else:
    #             # Use a 1x1 convolution to project channels to the desired number
    #             conv = nn.Conv2d(c, out_channels, kernel_size=1)
    #             x = conv(x)
    #
    #     # Adjust spatial dimensions: upsample by patch_size
    #     x = F.interpolate(x, scale_factor=patch_size, mode="nearest")
    #     # Now x will have shape: (batch, out_channels, h * patch_size, w * patch_size)
    #     return x
