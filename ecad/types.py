from typing import Any, TypeVar, TypedDict
from typing_extensions import NotRequired
import torch


class PromptEmbedding(TypedDict):
    pass


PromptEmbeddingType = TypeVar("PromptEmbeddingType", bound=PromptEmbedding)


class PixArtPromptEmbedding(PromptEmbedding):
    prompt_embeds: torch.Tensor
    prompt_attention_mask: torch.Tensor
    negative_prompt_embeds: torch.Tensor
    negative_prompt_attention_mask: torch.Tensor


class FluxPromptEmbedding(PromptEmbedding):
    """Embedding(s) for prompt(s) used by Flux.

    While flux can handle 2 different prompts, we always duplicate the same prompt to both encoders.
    Note that the `text_ids` attribute is encoded by Flux' encoders, but does NOT need to be dumped,
    as it is always recomputed during image generation anyways.

    Attributes:
        prompt_embeds: one or more prompt embeddings.
        pooled_prompt_embeds: one or more pooled prompt embeddings.
        text_ids: optional text ids.
    """

    prompt_embeds: torch.Tensor
    pooled_prompt_embeds: torch.Tensor
    text_ids: torch.Tensor | None


class PipelineConfig(TypedDict):
    name: str
    kwargs: dict[str, Any]


class ImageGeneratorConfig(TypedDict):
    pipeline: NotRequired[PipelineConfig]
    transformer_weights: NotRequired[str]
    pipeline_weights: NotRequired[str]
    image_generator: NotRequired[str]


class CustomFuncDict(TypedDict):
    name: str
    kwargs: dict[str, Any]


class ComponentScheduleDict(TypedDict):
    pass


class PixArtComponentScheduleDict(ComponentScheduleDict):
    attn1: bool
    attn2: bool
    ff: bool
    custom_compute_attn: NotRequired[CustomFuncDict]
    custom_compute_ff: NotRequired[CustomFuncDict]


class FluxSingleComponentScheduleDict(ComponentScheduleDict):
    single_attn: bool
    single_proj_mlp: bool
    single_proj_out: bool


class FluxFullComponentScheduleDict(ComponentScheduleDict):
    full_attn: bool
    full_ff: bool
    full_ff_context: bool


FluxComponentScheduleDict = (
    FluxSingleComponentScheduleDict | FluxFullComponentScheduleDict
)


ComponentScheduleDictT = TypeVar(
    "ComponentScheduleDictT", bound=ComponentScheduleDict
)

BlockScheduleDict = dict[str, ComponentScheduleDictT]
PixArtBlockScheduleDict = BlockScheduleDict[PixArtComponentScheduleDict]
FluxBlockScheduleDict = BlockScheduleDict[FluxComponentScheduleDict]

# maps inference steps (ints) to dict mapping block name to component schedules
CacheScheduleDict = dict[int, BlockScheduleDict]
PixArtCacheScheduleDict = dict[int, PixArtBlockScheduleDict]
FluxCacheScheduleDict = dict[int, FluxBlockScheduleDict]
