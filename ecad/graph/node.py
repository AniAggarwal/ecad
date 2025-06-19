from abc import ABC
from typing import Any, Callable

import torch
import torch.fx
from torch import nn

from ecad.graph.func_registry import FUNC_REGISTRY, DEFAULT_FUNC_NAME


class BaseTransformerNode(ABC):
    block_num: str

    def to_json(self) -> dict[str, Any]:
        return {"block_num": self.block_num}

    @classmethod
    def from_json(
        cls, json_dict: dict[str, Any], block: nn.Module | None = None
    ) -> "BaseTransformerNode":
        match json_dict["block_num"]:
            case "input":
                return InputTransformerNode.from_json(json_dict)
            case "output":
                return OutputTransformerNode.from_json(json_dict)
            case _ if "dummy" in json_dict["block_num"]:
                return DummyTransformerNode.from_json(json_dict)
            case _ if json_dict["block_num"].isdigit():
                return TransformerNode.from_json(json_dict, block)
            case _:
                raise ValueError(
                    f"Unknown block_num: {json_dict['block_num']}"
                )


class TransformerNode(BaseTransformerNode):

    def __init__(
        self,
        block_num: str,
        block: nn.Module,
        inputs: list[str],
        outputs: list[str],
        skip: bool = False,
        repeat_count: int = 0,
        repeat_target: str | None = None,
        input_type: str = DEFAULT_FUNC_NAME,
    ):

        self.block_num: str = block_num
        self.block: nn.Module = block

        self.inputs: list[str] = inputs
        self.outputs: list[str] = outputs
        self.skip: bool = skip

        self.repeat_count: int = repeat_count
        self.repeat_target: str | None = repeat_target

        self.input_type: str = input_type
        self.input_func: Callable[..., torch.Tensor] = FUNC_REGISTRY[
            self.input_type
        ]

    def to_json(self) -> dict[str, Any]:
        return {
            "block_num": self.block_num,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "skip": self.skip,
            "repeat_count": self.repeat_count,
            "repeat_target": self.repeat_target,
            "input_type": self.input_type,
        }

    @classmethod
    def from_json(
        cls, json_dict: dict[str, Any], block: nn.Module | None = None
    ) -> "TransformerNode":
        if block is None:
            raise ValueError("block must be provided for TransformerNode.")

        return cls(
            block_num=json_dict["block_num"],
            block=block,
            inputs=json_dict.get("inputs", []),
            outputs=json_dict.get("outputs", []),
            skip=json_dict.get("skip", False),
            repeat_count=json_dict.get("repeat_count", 0),
            repeat_target=json_dict.get("repeat_target", None),
            input_type=json_dict.get("input_type", DEFAULT_FUNC_NAME),
        )


class DummyTransformerNode(TransformerNode):

    def __init__(
        self,
        block_num: str,
        inputs: list[str],
        outputs: list[str],
        repeat_count: int = 0,
        repeat_target: str | None = None,
        input_type: str = DEFAULT_FUNC_NAME,
    ):
        super().__init__(
            block_num=block_num,
            block=nn.Identity(),  # identity for dummy nodes
            inputs=inputs,
            outputs=outputs,
            skip=True,  # always skip dummies
            repeat_count=repeat_count,
            repeat_target=repeat_target,
            input_type=input_type,
        )

    @classmethod
    def from_json(
        cls, json_dict: dict[str, Any], block: nn.Module | None = None
    ) -> "DummyTransformerNode":
        return cls(
            block_num=json_dict["block_num"],
            inputs=json_dict.get("inputs", []),
            outputs=json_dict.get("outputs", []),
            repeat_count=json_dict.get("repeat_count", 0),
            repeat_target=json_dict.get("repeat_target", None),
            input_type=json_dict.get("input_type", DEFAULT_FUNC_NAME),
        )


class InputTransformerNode(BaseTransformerNode):

    def __init__(self, outputs: list[str], block_num="input"):
        self.block_num = block_num
        self.outputs: list[str] = outputs

    def to_json(self) -> dict[str, Any]:
        return {
            "block_num": self.block_num,
            "outputs": self.outputs,
        }

    @classmethod
    def from_json(
        cls, json_dict: dict[str, Any], block=None
    ) -> "InputTransformerNode":
        return cls(outputs=json_dict.get("outputs", []))


class OutputTransformerNode(BaseTransformerNode):

    def __init__(
        self,
        inputs: list[str],
        input_type: str = DEFAULT_FUNC_NAME,
        block_num="output",
    ):
        self.block_num = block_num
        self.inputs: list[str] = inputs
        self.input_type: str = input_type
        self.input_func: Callable[..., torch.Tensor] = FUNC_REGISTRY[
            self.input_type
        ]

    def to_json(self) -> dict[str, Any]:
        return {
            "block_num": self.block_num,
            "inputs": self.inputs,
            "input_type": self.input_type,
        }

    @classmethod
    def from_json(
        cls, json_dict: dict[str, Any], block=None
    ) -> "OutputTransformerNode":
        return cls(
            inputs=json_dict.get("inputs", []),
            input_type=json_dict.get("input_type", DEFAULT_FUNC_NAME),
        )
