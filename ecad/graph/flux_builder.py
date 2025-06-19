import operator
from pathlib import Path
from typing import Iterable, Any

from graphviz import Digraph
import torch
import torch.fx
from torch import nn

from ecad.graph.builder import TransformerGraphBuilder, BuilderConfig
from ecad.graph.node import (
    InputTransformerNode,
    OutputTransformerNode,
    BaseTransformerNode,
)
from ecad.utils import ForwardArgs


class FluxTransformerGraphBuilder(TransformerGraphBuilder):

    def __init__(
        self,
        json_config: BuilderConfig,
        forward_args_type: type[ForwardArgs],
        num_blocks: int,
        num_single_blocks: int,
        transformer_blocks: nn.ModuleList | None = None,
        single_transformer_blocks: nn.ModuleList | None = None,
    ):
        self.json_config: BuilderConfig = json_config
        self.forward_args_type = forward_args_type

        # if transformer blocks not provided, use nn.Identity() placeholders
        if transformer_blocks is None:
            transformer_blocks = nn.ModuleList(
                (nn.Identity() for _ in range(num_blocks))
            )
        if single_transformer_blocks is None:
            single_transformer_blocks = nn.ModuleList(
                (nn.Identity() for _ in range(num_single_blocks))
            )

        self.verify_matching_io(self.json_config)
        self.check_for_cycles(self.json_config)

        root = self.register_transformer_blocks(
            transformer_blocks, single_transformer_blocks
        )
        graph = self.build_graph()
        torch.fx.GraphModule.__init__(self, root, graph)
        self.graph.lint()

    def register_transformer_blocks(
        self,
        transformer_blocks: nn.ModuleList | Iterable[nn.Module],
        single_transformer_blocks: nn.ModuleList | Iterable[nn.Module] = None,  # type: ignore
        **kwargs: Any,
    ) -> nn.Module:
        if single_transformer_blocks is None:
            raise ValueError(
                "single_transformer_blocks must be provided to FluxTransformerGraphBuilder."
            )

        # register transformer blocks to the root module
        root = nn.Module()
        self.transformer_block_map: dict[int, nn.Module] = {}
        self.single_transformer_block_map: dict[int, nn.Module] = {}

        for idx, block in enumerate(transformer_blocks):
            block_name = f"transformer_block_{idx}"
            setattr(root, block_name, block)
            self.transformer_block_map[idx] = block

        for idx, block in enumerate(single_transformer_blocks):
            block_name = f"single_transformer_block_{idx}"
            setattr(root, block_name, block)
            self.single_transformer_block_map[idx] = block

        return root

    def parse_config(
        self,
        json_config: BuilderConfig,
    ) -> dict[str, BaseTransformerNode]:
        # NOTE: this isn't used yet, since we force sequential forward pass
        raise NotImplementedError(
            "FluxTransformerGraphBuilder does not support parse_config or non sequential forward passes yet."
        )

    def build_graph(
        self,
        input_tnode: InputTransformerNode = None,
        output_tnode: OutputTransformerNode = None,
    ) -> torch.fx.Graph:
        # NOTE: this ALWAYS creates a normal forward pass in order for now (thus inputs are ignored)

        graph = torch.fx.Graph()
        self.graph_node_map: dict[str, torch.fx.Node] = {}

        # input placeholder, i.e. input to forward pass
        forward_args_node = graph.placeholder("forward_args")
        for arg in self.forward_args_type.fields():
            node = graph.create_node(
                "call_function",
                getattr,
                args=(forward_args_node, arg),
                name=f"forward_args.{arg}",
            )
            self.graph_node_map[f"forward_args.{arg}"] = node

        if not (
            "forward_args.hidden_states" in self.graph_node_map.keys()
            and "forward_args.encoder_hidden_states"
            in self.graph_node_map.keys()
        ):
            raise ValueError(
                "Hidden states and encoder hidden states must be provided in forward args."
            )

        # sequential forward pass over normal transformer blocks first
        for block_num in self.transformer_block_map.keys():
            # map arg names passed to transformer block forward -> graph nodes
            node_forward_args_kwargs: dict[str, torch.fx.Node] = {}
            for arg_name in self.forward_args_type.fields():
                if (
                    arg_name in ("hidden_states", "encoder_hidden_states")
                    and block_num != 0
                ):
                    full_name = f"{block_num - 1}"
                else:
                    full_name = f"forward_args"

                node_forward_args_kwargs[arg_name] = self.graph_node_map[
                    f"{full_name}.{arg_name}"
                ]

            node_name = f"{block_num}"
            self.graph_node_map[node_name] = graph.create_node(
                "call_module",
                f"transformer_block_{block_num}",
                kwargs=node_forward_args_kwargs,
                name=node_name,
            )

            # FluxTransformerBlock's return a tuple, so we need to unpack first
            # (NOTE: the order is encoder hiddens first)
            self.graph_node_map[f"{node_name}.encoder_hidden_states"] = (
                graph.create_node(
                    "call_function",
                    operator.getitem,
                    args=(self.graph_node_map[node_name], 0),
                    name=f"{node_name}.encoder_hidden_states",
                )
            )
            self.graph_node_map[f"{node_name}.hidden_states"] = (
                graph.create_node(
                    "call_function",
                    operator.getitem,
                    args=(self.graph_node_map[node_name], 1),
                    name=f"{node_name}.hidden_states",
                )
            )

        # after the normal transformer blocks, we must cat the encoder hiddens and hiddens
        # i.e.: hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
        final_block_num = list(self.transformer_block_map.keys())[-1]
        final_encoder_hidden_states_node = self.graph_node_map[
            f"{final_block_num}.encoder_hidden_states"
        ]
        final_hidden_states_node = self.graph_node_map[
            f"{final_block_num}.hidden_states"
        ]

        self.graph_node_map["single_forward_args.hidden_states"] = (
            graph.create_node(
                "call_function",
                torch.cat,
                args=(
                    [
                        final_encoder_hidden_states_node,
                        final_hidden_states_node,
                    ],
                ),
                kwargs={"dim": 1},
                name="single_forward_args.hidden_states",
            )
        )

        # we don't need encoder_hidden_states anymore
        fields_without_encoder = [
            x
            for x in self.forward_args_type.fields()
            if x != "encoder_hidden_states"
        ]

        # finally, sequential forward pass over single transformer blocks
        # for simplicity, we leave all forward args as forward_args.foo, except for
        # hidden states, whose cat'd version is now single_forward_args.hidden_states
        # And output of each transformer block now just single_{block_num}
        for block_num in self.single_transformer_block_map.keys():
            # map arg names passed to transformer block forward -> graph nodes
            node_forward_args_kwargs = {}
            for arg_name in fields_without_encoder:
                if arg_name == "hidden_states":
                    full_name = (
                        f"single_forward_args.hidden_states"
                        if block_num == 0
                        else f"single_{block_num - 1}"
                    )
                else:
                    full_name = f"forward_args.{arg_name}"

                node_forward_args_kwargs[arg_name] = self.graph_node_map[
                    full_name
                ]

            node_name = f"single_{block_num}"
            self.graph_node_map[node_name] = graph.create_node(
                "call_module",
                f"single_transformer_block_{block_num}",
                kwargs=node_forward_args_kwargs,
                name=node_name,
            )

        # now just the output graph node
        final_block_num = list(self.single_transformer_block_map.keys())[-1]
        self.graph_node_map["output"] = graph.output(
            self.graph_node_map[f"single_{final_block_num}"]
        )

        return graph

    def build_graph_bfs(
        self,
        graph: torch.fx.Graph,
        start_node: BaseTransformerNode,
        end_node: BaseTransformerNode,
        curr_suffix: int,
    ) -> int:
        raise NotImplementedError(
            "FluxTransformerGraphBuilder does not support build_graph_bfs yet."
        )

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
                if isinstance(arg, (list, tuple)):
                    for arg_elem in arg:
                        if isinstance(arg_elem, torch.fx.Node):
                            dot.edge(arg_elem.name, node.name)
            for kwarg in node.kwargs.values():
                if isinstance(kwarg, torch.fx.Node):
                    if not show_forward_args and "forward_args" in kwarg.name:
                        continue
                    dot.edge(kwarg.name, node.name)

        # Render and save graph
        dot.render(outfile=output_path, format="png", cleanup=True)
        print(f"Graph visualization saved to {output_path}.png")
