from diffusers.models.transformers.transformer_flux import (
    FluxTransformerBlock,
    FluxSingleTransformerBlock,
)
import torch

from ecad.schedulers.cache_scheduler.flux_cache_schedule import (
    FluxCacheSchedule,
)


class CachedFluxSingleTransformerBlock(FluxSingleTransformerBlock):
    def __init__(
        self,
        # for the cached version
        block_num: str,
        cache_schedule: FluxCacheSchedule,
        # below are passed onto the parent class
        dim,
        num_attention_heads,
        attention_head_dim,
        mlp_ratio=4.0,
        # Optional state dict to load weights
        state_dict=None,
    ):
        super().__init__(
            dim, num_attention_heads, attention_head_dim, mlp_ratio
        )

        if not block_num.startswith("single_"):
            raise ValueError(
                "block_num should have format 'single_X' for single block number X for CachedFluxSingleTransformerBlock"
            )

        if state_dict is not None:
            self.load_state_dict(state_dict, strict=True)

        self.block_num: str = block_num
        self.cache_schedule: FluxCacheSchedule = cache_schedule

        self.cached_attn_output: torch.Tensor | None = None
        self.cached_proj_mlp_output: torch.Tensor | None = None
        self.cached_proj_out_output: torch.Tensor | None = None

    def reset_cache(self) -> None:
        self.cached_attn_output = None
        self.cached_proj_mlp_output = None
        self.cached_proj_out_output = None

    def compute_attn_cached(
        self, hidden_states: torch.Tensor, image_rotary_emb: torch.Tensor
    ) -> torch.Tensor:
        recompute = self.cache_schedule.get_recompute(
            self.block_num, "single_attn"
        )
        no_cache = self.cached_attn_output is None

        if not recompute and no_cache:
            print(f"WARNING: No cached attn found. Recomputing.")

        if recompute or no_cache:
            attn_output = self.attn(
                hidden_states=hidden_states,
                image_rotary_emb=image_rotary_emb,
            )
        else:
            attn_output: torch.Tensor = self.cached_attn_output  # type: ignore

        # update cache
        self.cached_attn_output = attn_output
        return attn_output

    def compute_proj_cached(
        self, component: str, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        if component not in ["proj_mlp", "proj_out"]:
            raise ValueError(
                f"component should be one of ['proj_mlp', 'proj_out'] but got {component}"
            )

        recompute = self.cache_schedule.get_recompute(
            self.block_num, f"single_{component}"
        )
        cache_attr = f"cached_{component}_output"
        no_cache = getattr(self, cache_attr) is None

        if not recompute and no_cache:
            print(f"WARNING: No cached {component} found. Recomputing.")

        if recompute or no_cache:
            proj_output = getattr(self, component)(hidden_states)
        else:
            proj_output: torch.Tensor = getattr(self, cache_attr)

        # update cache
        setattr(self, cache_attr, proj_output)
        return proj_output

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        temb: torch.FloatTensor,
        image_rotary_emb=None,
    ):
        residual = hidden_states
        norm_hidden_states, gate = self.norm(hidden_states, emb=temb)

        # NOTE: replaced call to self.proj_mlp with self.compute_pro
        mlp_hidden_states = self.act_mlp(
            self.compute_proj_cached("proj_mlp", norm_hidden_states)
        )

        # NOTE: replaced call to self.attn with self.compute_attn
        attn_output = self.compute_attn_cached(
            hidden_states=norm_hidden_states,
            image_rotary_emb=image_rotary_emb,
        )

        hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=2)
        gate = gate.unsqueeze(1)

        # NOTE: replaced call to self.proj_out with self.compute_pro
        hidden_states = gate * self.compute_proj_cached(
            "proj_out", hidden_states
        )
        hidden_states = residual + hidden_states
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)

        return hidden_states


class CachedFluxTransformerBlock(FluxTransformerBlock):

    def __init__(
        self,
        # for the cached version
        block_num: str,
        cache_schedule: FluxCacheSchedule,
        # below are passed onto the parent class
        dim,
        num_attention_heads,
        attention_head_dim,
        qk_norm="rms_norm",
        eps=1e-6,
        # Optional state dict to load weights
        state_dict=None,
    ):
        super().__init__(
            dim, num_attention_heads, attention_head_dim, qk_norm, eps
        )

        if state_dict is not None:
            self.load_state_dict(state_dict, strict=True)

        self.block_num: str = block_num
        self.cache_schedule: FluxCacheSchedule = cache_schedule

        self.cached_attn_output: torch.Tensor | None = None
        self.cached_context_attn_output: torch.Tensor | None = None
        self.cached_ff_output: torch.Tensor | None = None
        self.cached_ff_context_output: torch.Tensor | None = None

    def reset_cache(self) -> None:
        self.cached_attn_output = None
        self.cached_context_attn_output = None
        self.cached_ff_output = None
        self.cached_ff_context_output = None

    def compute_attn_cached(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        image_rotary_emb: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        recompute = self.cache_schedule.get_recompute(
            self.block_num, "full_attn"
        )
        no_cache = (
            self.cached_attn_output is None
            or self.cached_context_attn_output is None
        )

        if not recompute and no_cache:
            print(f"WARNING: No cached attn found. Recomputing.")

        if recompute or no_cache:
            attn_output, context_attn_output = self.attn(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                image_rotary_emb=image_rotary_emb,
            )
        else:
            attn_output: torch.Tensor = self.cached_attn_output  # type: ignore
            context_attn_output: torch.Tensor = self.cached_context_attn_output  # type: ignore

        # update cache
        self.cached_attn_output = attn_output
        self.cached_context_attn_output = context_attn_output
        return attn_output, context_attn_output

    def compute_ff_cached(
        self, component: str, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        if component not in ["ff", "ff_context"]:
            raise ValueError(
                f"component should be one of ['ff', 'ff_context'] but got {component}"
            )

        recompute = self.cache_schedule.get_recompute(
            self.block_num, f"full_{component}"
        )
        cache_attr = f"cached_{component}_output"
        no_cache = getattr(self, cache_attr) is None

        if not recompute and no_cache:
            print(f"WARNING: No cached {component} found. Recomputing.")

        if recompute or no_cache:
            proj_output = getattr(self, component)(hidden_states)
        else:
            proj_output: torch.Tensor = getattr(self, cache_attr)

        # update cache
        setattr(self, cache_attr, proj_output)
        return proj_output

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor,
        temb: torch.FloatTensor,
        image_rotary_emb=None,
    ):
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.norm1(hidden_states, emb=temb)
        )

        (
            norm_encoder_hidden_states,
            c_gate_msa,
            c_shift_mlp,
            c_scale_mlp,
            c_gate_mlp,
        ) = self.norm1_context(encoder_hidden_states, emb=temb)

        # Attention.
        # NOTE: replaced call to self.attn with self.compute_attn
        attn_output, context_attn_output = self.compute_attn_cached(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
        )

        # Process attention outputs for the `hidden_states`.
        attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = hidden_states + attn_output

        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = (
            norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        )

        # NOTE: replaced call to self.ff with self.compute_ff_cached
        ff_output = self.compute_ff_cached("ff", norm_hidden_states)
        ff_output = gate_mlp.unsqueeze(1) * ff_output

        hidden_states = hidden_states + ff_output

        # Process attention outputs for the `encoder_hidden_states`.

        context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
        encoder_hidden_states = encoder_hidden_states + context_attn_output

        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
        norm_encoder_hidden_states = (
            norm_encoder_hidden_states * (1 + c_scale_mlp[:, None])
            + c_shift_mlp[:, None]
        )

        # NOTE: replaced call to self.ff_context with self.compute_ff_cached
        context_ff_output = self.compute_ff_cached(
            "ff_context", norm_encoder_hidden_states
        )
        encoder_hidden_states = (
            encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output
        )
        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)

        return encoder_hidden_states, hidden_states
