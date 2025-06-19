from pathlib import Path
from typing import Iterable, Any

import torch
import torch.fx
from graphviz import Digraph
from torch import nn

from ecad.graph.builder import TransformerGraphBuilder, BuilderConfig
from ecad.graph.func_registry import FUNC_REGISTRY
from ecad.graph.node import (
    InputTransformerNode,
    OutputTransformerNode,
    BaseTransformerNode,
    TransformerNode,
    DummyTransformerNode,
)
from ecad.utils import ForwardArgs


class PixArtTransformerGraphBuilder(TransformerGraphBuilder):

    def __init__(
        self,
        json_config: BuilderConfig,
        forward_args_type: type[ForwardArgs],
        transformer_blocks: nn.ModuleList | None = None,
    ):
        self.json_config: BuilderConfig = json_config
        self.forward_args_type = forward_args_type

        # count number of transformer blocks if not provided
        # and use nn.Identity() placeholders for each block
        if transformer_blocks is None:
            num_blocks = 0
            for block_num in json_config:
                if block_num.isdigit():
                    num_blocks += 1
            transformer_blocks = nn.ModuleList(
                (nn.Identity() for _ in range(num_blocks))
            )

        self.verify_matching_io(self.json_config)
        self.check_for_cycles(self.json_config)

        root = self.register_transformer_blocks(transformer_blocks)
        self.transformer_node_map = self.parse_config(json_config)
        self.input_tnode: InputTransformerNode = self.transformer_node_map["input"]  # type: ignore
        self.output_tnode: OutputTransformerNode = self.transformer_node_map["output"]  # type: ignore

        graph = self.build_graph(
            self.input_tnode,
            self.output_tnode,
        )
        # super().__init__(root, graph)
        torch.fx.GraphModule.__init__(self, root, graph)
        self.graph.lint()

    def register_transformer_blocks(
        self,
        transformer_blocks: nn.ModuleList | Iterable[nn.Module],
        **kwargs: Any,
    ) -> nn.Module:
        # register transformer blocks to the root module
        root = nn.Module()
        self.transformer_block_map = {}
        for idx, block in enumerate(transformer_blocks):
            block_name = f"transformer_block_{idx}"
            setattr(root, block_name, block)
            self.transformer_block_map[idx] = block

        return root

    def parse_config(
        self,
        json_config: BuilderConfig,
    ) -> dict[str, BaseTransformerNode]:
        nodes: dict[str, BaseTransformerNode] = {}

        for block_num, block_config in json_config.items():
            block = (
                self.transformer_block_map[int(block_num)]
                if block_num.isdigit()
                else None
            )
            nodes[block_num] = BaseTransformerNode.from_json(
                block_config | {"block_num": block_num}, block
            )

        if "input" not in nodes or "output" not in nodes:
            raise ValueError(
                "Input and output nodes must be defined in the graph config."
            )
        return nodes

    def build_graph(
        self,
        input_tnode: InputTransformerNode,
        output_tnode: OutputTransformerNode,
    ) -> torch.fx.Graph:
        graph = torch.fx.Graph()
        self.graph_node_map: dict[str, torch.fx.Node] = {}

        # input placeholder
        forward_args_node = graph.placeholder("forward_args")
        for arg in self.forward_args_type.fields():
            node = graph.create_node(
                "call_function",
                getattr,
                args=(forward_args_node, arg),
                name=f"forward_args.{arg}",
            )
            self.graph_node_map[f"forward_args.{arg}"] = node

        suffix = 0
        # rename the hidden states to the input graph node
        self.graph_node_map[f"input:{suffix}"] = self.graph_node_map[
            "forward_args.hidden_states"
        ]
        self.graph_node_map.pop("forward_args.hidden_states")

        self.build_graph_bfs(graph, input_tnode, output_tnode, suffix)

        return graph

    def build_graph_bfs(
        self,
        graph: torch.fx.Graph,
        start_node: BaseTransformerNode,
        end_node: BaseTransformerNode,
        curr_suffix: int,
    ) -> int:
        # Note that the start node should already be in the graph_node_map
        queue: list[BaseTransformerNode] = [start_node]
        visited = set()

        while queue:
            curr_node = queue.pop(0)

            if curr_node in visited:
                continue
            visited.add(curr_node)

            if isinstance(curr_node, (OutputTransformerNode, TransformerNode)):

                # subtract 1 only for the start node as it uses the previous suffix as input
                input_nodes = [
                    self.graph_node_map[
                        f"{node}:{curr_suffix - int(curr_node == start_node)}"
                    ]
                    for node in curr_node.inputs
                ]

                node_name = f"{curr_node.block_num}:{curr_suffix}.input_func"
                self.graph_node_map[node_name] = graph.create_node(
                    "call_function",
                    curr_node.input_func,
                    args=tuple(input_nodes),
                    name=node_name,
                )

            if isinstance(curr_node, OutputTransformerNode):
                self.graph_node_map[f"output:{curr_suffix}"] = graph.output(
                    self.graph_node_map[f"output:{curr_suffix}.input_func"]
                )

            if isinstance(curr_node, TransformerNode):
                node_name = f"{curr_node.block_num}:{curr_suffix}"
                if curr_node.skip:
                    # dummy nodes always have skip set to True
                    if isinstance(curr_node, DummyTransformerNode):
                        name = curr_node.block_num
                    else:
                        name = f"skip transformer_block_{curr_node.block_num}"

                    self.graph_node_map[node_name] = graph.create_node(
                        "call_function",
                        FUNC_REGISTRY["identity"],
                        args=(
                            self.graph_node_map[
                                f"{curr_node.block_num}:{curr_suffix}.input_func"
                            ],
                        ),
                        name=name,
                    )

                else:
                    node_forward_args_kwargs = {}

                    for arg_name in self.forward_args_type.fields():
                        if arg_name == "hidden_states":
                            str_name = f"{curr_node.block_num}:{curr_suffix}.input_func"
                        else:
                            str_name = f"forward_args.{arg_name}"
                        node_forward_args_kwargs[arg_name] = (
                            self.graph_node_map[str_name]
                        )

                    self.graph_node_map[node_name] = graph.create_node(
                        "call_module",
                        f"transformer_block_{curr_node.block_num}",
                        kwargs=node_forward_args_kwargs,
                        name=node_name,
                    )

                if curr_node.repeat_count >= 1:
                    assert (
                        curr_node.repeat_target is not None
                    ), "Repeat target not found."

                    curr_node.repeat_count -= 1

                    # update the input to the repeat target to the current node
                    # and its input func to be the identity
                    target_tnode: TransformerNode = self.transformer_node_map[curr_node.repeat_target]  # type: ignore
                    target_tnode.inputs = [curr_node.block_num]
                    target_tnode.input_func = FUNC_REGISTRY["identity"]

                    # bfs recursively on the target node until we reach this curr_node again
                    curr_suffix = self.build_graph_bfs(
                        graph,
                        target_tnode,
                        curr_node,
                        curr_suffix + 1,
                    )

                    curr_node.repeat_count += 1

            # once we've reached the end, don't add neighbors or any other processing
            if curr_node == end_node:
                break

            # enqueue neighbors
            if isinstance(curr_node, (TransformerNode, InputTransformerNode)):
                for output in curr_node.outputs:
                    queue.append(self.transformer_node_map[output])

        return curr_suffix

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
