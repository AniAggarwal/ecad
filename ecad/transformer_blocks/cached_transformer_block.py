from typing import Any, Dict, Optional
from diffusers.models.attention import (
    BasicTransformerBlock,
)
import torch

from ecad.schedulers.cache_scheduler.pixart_cache_schedule import (
    PixArtCacheSchedule,
)
from ecad.transformer_blocks.custom_attn_ff import (
    ComputeAttnRegistry,
    ComputeFFRegistry,
)

from diffusers.models.attention import (
    _chunked_feed_forward,
)


class CachedTransformerBlock(BasicTransformerBlock):
    r"""
    A cached version of diffusers' basic Transformer block.

    Parameters:
        dim (`int`): The number of channels in the input and output.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        cross_attention_dim (`int`, *optional*): The size of the encoder_hidden_states vector for cross attention.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to be used in feed-forward.
        num_embeds_ada_norm (:
            obj: `int`, *optional*): The number of diffusion steps used during training. See `Transformer2DModel`.
        attention_bias (:
            obj: `bool`, *optional*, defaults to `False`): Configure if the attentions should contain a bias parameter.
        only_cross_attention (`bool`, *optional*):
            Whether to use only cross-attention layers. In this case two cross attention layers are used.
        double_self_attention (`bool`, *optional*):
            Whether to use two self-attention layers. In this case no cross attention layers are used.
        upcast_attention (`bool`, *optional*):
            Whether to upcast the attention computation to float32. This is useful for mixed precision training.
        norm_elementwise_affine (`bool`, *optional*, defaults to `True`):
            Whether to use learnable elementwise affine parameters for normalization.
        norm_type (`str`, *optional*, defaults to `"layer_norm"`):
            The normalization layer to use. Can be `"layer_norm"`, `"ada_norm"` or `"ada_norm_zero"`.
        final_dropout (`bool` *optional*, defaults to False):
            Whether to apply a final dropout after the last feed-forward layer.
        attention_type (`str`, *optional*, defaults to `"default"`):
            The type of attention to use. Can be `"default"` or `"gated"` or `"gated-text-image"`.
        positional_embeddings (`str`, *optional*, defaults to `None`):
            The type of positional embeddings to apply to.
        num_positional_embeddings (`int`, *optional*, defaults to `None`):
            The maximum number of positional embeddings to apply.

    Special CachedTransformerBlock Parameters:
        cache_schedule (`CacheSchedule`): The schedule for when to cache the outputs of the attention and feed-forward layers.
    """

    def __init__(
        self,
        # for the cached version
        block_num: str,
        cache_schedule: PixArtCacheSchedule,
        # below are passed onto the parent class
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout=0.0,
        cross_attention_dim: Optional[int] = None,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        attention_bias: bool = False,
        only_cross_attention: bool = False,
        double_self_attention: bool = False,
        upcast_attention: bool = False,
        norm_elementwise_affine: bool = True,
        norm_type: str = "layer_norm",  # 'layer_norm', 'ada_norm', 'ada_norm_zero', 'ada_norm_single', 'ada_norm_continuous', 'layer_norm_i2vgen'
        norm_eps: float = 1e-5,
        final_dropout: bool = False,
        attention_type: str = "default",
        positional_embeddings: Optional[str] = None,
        num_positional_embeddings: Optional[int] = None,
        ada_norm_continous_conditioning_embedding_dim: Optional[int] = None,
        ada_norm_bias: Optional[int] = None,
        ff_inner_dim: Optional[int] = None,
        ff_bias: bool = True,
        attention_out_bias: bool = True,
    ):
        super().__init__(
            dim=dim,
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            dropout=dropout,
            cross_attention_dim=cross_attention_dim,
            activation_fn=activation_fn,
            num_embeds_ada_norm=num_embeds_ada_norm,
            attention_bias=attention_bias,
            only_cross_attention=only_cross_attention,
            double_self_attention=double_self_attention,
            upcast_attention=upcast_attention,
            norm_elementwise_affine=norm_elementwise_affine,
            norm_type=norm_type,
            norm_eps=norm_eps,
            final_dropout=final_dropout,
            attention_type=attention_type,
            positional_embeddings=positional_embeddings,
            num_positional_embeddings=num_positional_embeddings,
            ada_norm_continous_conditioning_embedding_dim=ada_norm_continous_conditioning_embedding_dim,
            ada_norm_bias=ada_norm_bias,
            ff_inner_dim=ff_inner_dim,
            ff_bias=ff_bias,
            attention_out_bias=attention_out_bias,
        )
        self.block_num: str = block_num
        self.cache_schedule: PixArtCacheSchedule = cache_schedule

        self.cached_attn1_output: torch.Tensor | None = None
        self.cached_attn2_output: torch.Tensor | None = None
        self.cached_ff_output: torch.Tensor | None = None

    def reset_cache(self) -> None:
        self.cached_attn1_output = None
        self.cached_attn2_output = None
        self.cached_ff_output = None

    def compute_attn(
        self,
        attn: str,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None,
        attention_mask: torch.Tensor | None,
        **cross_attention_kwargs,
    ) -> torch.Tensor:
        func_config = self.cache_schedule.get_custom_compute_attn(
            self.block_num
        )
        func_name = func_config.get("name", None)
        func_kwargs = func_config.get("kwargs", {})
        func = ComputeAttnRegistry.get(func_name, False)
        assert func is not None, f"Function {func_name} not found in registry"

        return func(
            self,
            attn,
            hidden_states,
            encoder_hidden_states,
            attention_mask,
            **func_kwargs,
            **cross_attention_kwargs,
        )

    def compute_ff(
        self,
        norm_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        func_config = self.cache_schedule.get_custom_compute_ff(self.block_num)
        func_name = func_config.get("name", None)
        func_kwargs = func_config.get("kwargs", {})
        func = ComputeFFRegistry.get(func_name, False)
        assert func is not None, f"Function {func_name} not found in registry"

        return func(
            self,
            norm_hidden_states,
            **func_kwargs,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        class_labels: Optional[torch.LongTensor] = None,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:

        if cross_attention_kwargs is not None:
            if cross_attention_kwargs.get("scale", None) is not None:
                # NOTE: replaced huggingface internal logging warning with print
                print(
                    "WARNING: Passing `scale` to `cross_attention_kwargs` is deprecated. `scale` will be ignored."
                )

        # Notice that normalization is always applied before the real computation in the following blocks.
        # 0. Self-Attention
        batch_size = hidden_states.shape[0]

        if self.norm_type == "ada_norm":
            norm_hidden_states = self.norm1(hidden_states, timestep)
        elif self.norm_type == "ada_norm_zero":
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.norm1(
                    hidden_states,
                    timestep,
                    class_labels,
                    hidden_dtype=hidden_states.dtype,
                )
            )
        elif self.norm_type in ["layer_norm", "layer_norm_i2vgen"]:
            norm_hidden_states = self.norm1(hidden_states)
        elif self.norm_type == "ada_norm_continuous":
            norm_hidden_states = self.norm1(
                hidden_states, added_cond_kwargs["pooled_text_emb"]
            )
        elif self.norm_type == "ada_norm_single":
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.scale_shift_table[None]
                + timestep.reshape(batch_size, 6, -1)
            ).chunk(6, dim=1)
            norm_hidden_states = self.norm1(hidden_states)
            norm_hidden_states = (
                norm_hidden_states * (1 + scale_msa) + shift_msa
            )
        else:
            raise ValueError("Incorrect norm used")

        if self.pos_embed is not None:
            norm_hidden_states = self.pos_embed(norm_hidden_states)

        # 1. Prepare GLIGEN inputs
        cross_attention_kwargs = (
            cross_attention_kwargs.copy()
            if cross_attention_kwargs is not None
            else {}
        )
        gligen_kwargs = cross_attention_kwargs.pop("gligen", None)

        # NOTE: replaced call to self.attn1 with this call
        attn_output = self.compute_attn(
            "attn1",
            norm_hidden_states,
            encoder_hidden_states=(
                encoder_hidden_states if self.only_cross_attention else None
            ),
            attention_mask=attention_mask,
            **cross_attention_kwargs,
        )

        if self.norm_type == "ada_norm_zero":
            attn_output = gate_msa.unsqueeze(1) * attn_output
        elif self.norm_type == "ada_norm_single":
            attn_output = gate_msa * attn_output

        hidden_states = attn_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        # 1.2 GLIGEN Control
        if gligen_kwargs is not None:
            hidden_states = self.fuser(hidden_states, gligen_kwargs["objs"])

        # 3. Cross-Attention
        if self.attn2 is not None:
            if self.norm_type == "ada_norm":
                norm_hidden_states = self.norm2(hidden_states, timestep)
            elif self.norm_type in [
                "ada_norm_zero",
                "layer_norm",
                "layer_norm_i2vgen",
            ]:
                norm_hidden_states = self.norm2(hidden_states)
            elif self.norm_type == "ada_norm_single":
                # For PixArt norm2 isn't applied here:
                # https://github.com/PixArt-alpha/PixArt-alpha/blob/0f55e922376d8b797edd44d25d0e7464b260dcab/diffusion/model/nets/PixArtMS.py#L70C1-L76C103
                norm_hidden_states = hidden_states
            elif self.norm_type == "ada_norm_continuous":
                norm_hidden_states = self.norm2(
                    hidden_states, added_cond_kwargs["pooled_text_emb"]
                )
            else:
                raise ValueError("Incorrect norm")

            if (
                self.pos_embed is not None
                and self.norm_type != "ada_norm_single"
            ):
                norm_hidden_states = self.pos_embed(norm_hidden_states)

            # NOTE: replaced call to self.attn2 with this call
            attn_output = self.compute_attn(
                "attn2",
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                **cross_attention_kwargs,
            )
            hidden_states = attn_output + hidden_states

        # 4. Feed-forward
        # i2vgen doesn't have this norm 🤷‍♂️
        if self.norm_type == "ada_norm_continuous":
            norm_hidden_states = self.norm3(
                hidden_states, added_cond_kwargs["pooled_text_emb"]
            )
        elif not self.norm_type == "ada_norm_single":
            norm_hidden_states = self.norm3(hidden_states)

        if self.norm_type == "ada_norm_zero":
            norm_hidden_states = (
                norm_hidden_states * (1 + scale_mlp[:, None])
                + shift_mlp[:, None]
            )

        if self.norm_type == "ada_norm_single":
            norm_hidden_states = self.norm2(hidden_states)
            norm_hidden_states = (
                norm_hidden_states * (1 + scale_mlp) + shift_mlp
            )

        # NOTE: moved if/else block here to within compute_ff (and added caching)
        ff_output = self.compute_ff(norm_hidden_states)

        if self.norm_type == "ada_norm_zero":
            ff_output = gate_mlp.unsqueeze(1) * ff_output
        elif self.norm_type == "ada_norm_single":
            ff_output = gate_mlp * ff_output

        hidden_states = ff_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        return hidden_states

    @ComputeAttnRegistry.register
    def compute_attn_cached(
        self,
        attn: str,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None,
        attention_mask: torch.Tensor | None,
        **cross_attention_kwargs,
    ) -> torch.Tensor:
        if attn not in ["attn1", "attn2"]:
            raise ValueError(
                f"Invalid attention type: {attn}. Must be attn1 or attn2"
            )

        recompute = self.cache_schedule.get_recompute(self.block_num, attn)
        cache_attr = f"cached_{attn}_output"
        no_cache = getattr(self, cache_attr) is None

        if not recompute and no_cache:
            print(f"WARNING: No cached {attn} found. Recomputing.")

        if recompute or no_cache:
            attn_output = getattr(self, attn)(
                hidden_states,
                encoder_hidden_states,
                attention_mask,
                **cross_attention_kwargs,
            )
        else:
            attn_output = getattr(self, cache_attr)

        # update the cache
        setattr(self, cache_attr, attn_output)

        return attn_output

    @ComputeFFRegistry.register
    def compute_ff_cached(
        self,
        norm_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        recompute = self.cache_schedule.get_recompute(self.block_num, "ff")
        no_cache = self.cached_ff_output is None

        if not recompute and no_cache:
            print(f"WARNING: No cached ff found. Recomputing.")

        if recompute or no_cache:
            if self._chunk_size is not None:
                ff_output = _chunked_feed_forward(
                    self.ff,
                    norm_hidden_states,
                    self._chunk_dim,
                    self._chunk_size,
                )
            else:
                ff_output: torch.Tensor = self.ff(norm_hidden_states)
        else:
            # shouldn't be needed, type checker is bugging out
            assert self.cached_ff_output is not None
            ff_output = self.cached_ff_output

        # update the cache
        self.cached_ff_output = ff_output

        return ff_output

    @ComputeAttnRegistry.register
    def compute_attn_tgate(
        self,
        attn: str,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None,
        attention_mask: torch.Tensor | None,
        gate_step: int | None = None,
        **cross_attention_kwargs,
    ) -> torch.Tensor:
        if attn not in ["attn1", "attn2"]:
            raise ValueError(
                f"Invalid attention type: {attn}. Must be attn1 or attn2"
            )

        # need to remove gate_step from cross_attention_kwargs if it ended up in there
        # since we don't want it passed to the attention layer
        gate_step = cross_attention_kwargs.pop("gate_step", gate_step)
        if gate_step is None:
            raise ValueError(
                "gate_step must be provided as a kwarg to commpute_attn_tgate."
            )

        # treat self attention like normal
        if attn == "attn1":
            return self.compute_attn_cached(
                attn,
                hidden_states,
                encoder_hidden_states,
                attention_mask,
                **cross_attention_kwargs,
            )

        # before the gate step, we treat cross attention like normal
        if self.cache_schedule.curr_step <= gate_step - 1:
            # note this updates the cahce, which we overwrite later
            hidden_states = self.compute_attn_cached(
                attn,
                hidden_states,
                encoder_hidden_states,
                attention_mask,
                **cross_attention_kwargs,
            )
        else:
            # on and after gate_step, we always use the cached version
            assert (
                self.cached_attn2_output is not None
            ), "Cross-Attention must be cached at gate step for TGATE."
            hidden_states = self.cached_attn2_output

        if self.cache_schedule.curr_step == gate_step - 1:
            # we need to cache an averaged version of the cross attention
            # but return the unaveraged version
            hidden_uncond, hidden_pred_text = hidden_states.chunk(2)
            to_cache = (hidden_uncond + hidden_pred_text) / 2
        else:
            to_cache = hidden_states

        # update the cache
        self.cached_attn2_output = to_cache

        return hidden_states
