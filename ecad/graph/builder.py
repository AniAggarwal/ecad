from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Iterable
import torch
import torch.fx
from torch.fx import GraphModule
from torch import nn
from graphviz import Digraph

from ecad.graph.node import (
    BaseTransformerNode,
    InputTransformerNode,
    OutputTransformerNode,
)
from ecad.utils import ForwardArgs

# TODO:
# - add some check for len of transformer blocks and json config

BuilderConfig = dict[str, dict[str, Any]]


class TransformerGraphBuilder(GraphModule, ABC):

    @abstractmethod
    def __init__(
        self,
        json_config: BuilderConfig,
        forward_args_type: type[ForwardArgs],
        transformer_blocks: nn.ModuleList | None = None,
        **kwargs: Any,
    ):
        pass

    @abstractmethod
    def register_transformer_blocks(
        self,
        transformer_blocks: nn.ModuleList | Iterable[nn.Module],
        **kwargs: Any,
    ) -> nn.Module:
        pass

    def update_transformer_blocks(
        self,
        transformer_blocks: nn.ModuleList | Iterable[nn.Module],
        **kwargs: Any,
    ) -> None:
        root = self.register_transformer_blocks(transformer_blocks, **kwargs)
        torch.fx.GraphModule.__init__(self, root, self.graph)
        self.graph.lint()

    @abstractmethod
    def parse_config(
        self,
        json_config: BuilderConfig,
    ) -> dict[str, BaseTransformerNode]:
        pass

    @abstractmethod
    def build_graph(
        self,
        input_tnode: InputTransformerNode,
        output_tnode: OutputTransformerNode,
    ) -> torch.fx.Graph:
        pass

    @abstractmethod
    def build_graph_bfs(
        self,
        graph: torch.fx.Graph,
        start_node: BaseTransformerNode,
        end_node: BaseTransformerNode,
        curr_suffix: int,
    ) -> int:
        pass

    def visualize_fx_graph(
        self, output_path: Path, show_forward_args: bool = False
    ):
        dot = Digraph()

        # Add nodes
        for node in self.graph.nodes:
            label = (
                f"{node.target}\n" if node.target is not None else ""
            ) + f"{node.op}: {node.name}"
            dot.node(node.name, label)

        # Add edges
        for node in self.graph.nodes:
            for arg in node.args:
                if isinstance(arg, torch.fx.Node):
                    dot.edge(arg.name, node.name)
            for kwarg in node.kwargs.values():
                if isinstance(kwarg, torch.fx.Node):
                    if not show_forward_args and "forward_args" in kwarg.name:
                        continue
                    dot.edge(kwarg.name, node.name)

        # Render and save graph
        dot.render(outfile=output_path, format="png", cleanup=True)
        print(f"Graph visualization saved to {output_path}.png")

    def to_json(self) -> dict[str, Any]:
        return self.json_config

    @staticmethod
    def verify_matching_io(json_config: BuilderConfig):
        all_inputs = set()
        all_outputs = set()

        for name, node_conf in json_config.items():

            # must have aggregate function if multiple inputs
            if (
                "inputs" in node_conf
                and len(node_conf["inputs"]) > 1
                and "input_type" not in node_conf
            ):
                raise ValueError(
                    f"Node {name} has multiple inputs but no input_type defined."
                )

            for inpt in node_conf.get("inputs", []):
                all_inputs.add(inpt)

                if inpt not in json_config:
                    raise ValueError(
                        f"Node {name} has input {inpt} but is missing from the graph."
                    )

                outs = json_config[inpt].get("outputs", [])
                if name not in outs:
                    raise ValueError(
                        f"Node {name} has input {inpt} but missing from {inpt}.outputs: {outs}."
                    )

            for output in node_conf.get("outputs", []):
                all_outputs.add(output)

                if output not in json_config:
                    raise ValueError(
                        f"Node {name} has output {output} but is missing from the graph."
                    )

                inpts = json_config[output].get("inputs", [])
                if name not in inpts:
                    raise ValueError(
                        f"Node {name} has output: {output}, but is missing from {output}.inputs: {inpts}."
                    )
        # TODO: this raises false positives. need to debug at some point
        # if not (
        #     all_inputs.symmetric_difference(all_outputs) == {"input", "output"}
        #     and all_inputs == json_config.keys()
        #     and all_outputs == json_config.keys()
        # ):
        #     raise ValueError(f"Inputs, outputs, and nodes do not all match.")

    @staticmethod
    def check_for_cycles(json_config: BuilderConfig):
        visited = set()
        stack = set()

        def dfs(node):
            if node in stack:
                raise ValueError("Cycle detected in graph configuration.")
            if node not in visited:
                stack.add(node)
                for neighbor in json_config[node].get("outputs", []):
                    dfs(neighbor)
                stack.remove(node)
                visited.add(node)

        for node in json_config:
            dfs(node)
