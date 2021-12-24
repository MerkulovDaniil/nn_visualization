from __future__ import absolute_import, division, print_function
import os
import re
from random import getrandbits
import inspect
import numpy as np
import copy

import torch
import torch.nn as nn


from collections import OrderedDict
import numpy as np


class NodePattern():
    def __init__(self, op, condition=None):
        self.op = op
        self.condition = condition  # TODO: not implemented yet

    def match(self, graph, node):
        if isinstance(node, list):
            return [], None
        if self.op == node.op:
            following = graph.outgoing(node)
            if len(following) == 1:
                following = following[0]
            return [node], following
        else:
            return [], None


class SerialPattern():
    def __init__(self, patterns):
        self.patterns = patterns

    def match(self, graph, node):
        all_matches = []
        for i, p in enumerate(self.patterns):
            matches, following = p.match(graph, node)
            if not matches:
                return [], None
            all_matches.extend(matches)
            if i < len(self.patterns) - 1:
                node = following  # Might be more than one node
        return all_matches, following


class ParallelPattern():
    def __init__(self, patterns):
        self.patterns = patterns

    def match(self, graph, nodes):
        if not nodes:
            return [], None
        nodes = nodes if isinstance(nodes, list) else [nodes]
        # If a single node, assume we need to match with its siblings
        if len(nodes) == 1:
            nodes = graph.siblings(nodes[0])
        else:
            # Verify all nodes have the same parent or all have no parent
            parents = [graph.incoming(n) for n in nodes]
            matches = [set(p) == set(parents[0]) for p in parents[1:]]
            if not all(matches):
                return [], None

        # TODO: If more nodes than patterns, we should consider
        #       all permutations of the nodes
        if len(self.patterns) != len(nodes):
            return [], None

        patterns = self.patterns.copy()
        nodes = nodes.copy()
        all_matches = []
        end_node = None
        for p in patterns:
            found = False
            for n in nodes:
                matches, following = p.match(graph, n)
                if matches:
                    found = True
                    nodes.remove(n)
                    all_matches.extend(matches)
                    # Verify all branches end in the same node
                    if end_node:
                        if end_node != following:
                            return [], None
                    else:
                        end_node = following
                    break
            if not found:
                return [], None
        return all_matches, end_node


class GEParser():
    def __init__(self, text):
        self.index = 0
        self.text = text

    def parse(self):
        return self.serial() or self.parallel() or self.expression()

    def parallel(self):
        index = self.index
        expressions = []
        while len(expressions) == 0 or self.token("|"):
            e = self.expression()
            if not e:
                break
            expressions.append(e)
        if len(expressions) >= 2:
            return ParallelPattern(expressions)
        # No match. Reset index
        self.index = index

    def serial(self):
        index = self.index
        expressions = []
        while len(expressions) == 0 or self.token(">"):
            e = self.expression()
            if not e:
                break
            expressions.append(e)

        if len(expressions) >= 2:
            return SerialPattern(expressions)
        self.index = index

    def expression(self):
        index = self.index

        if self.token("("):
            e = self.serial() or self.parallel() or self.op()
            if e and self.token(")"):
                return e
        self.index = index
        e = self.op()
        return e

    def op(self):
        t = self.re(r"\w+")
        if t:
            c = self.condition()
            return NodePattern(t, c)

    def condition(self):
        # TODO: not implemented yet. This function is a placeholder
        index = self.index
        if self.token("["):
            c = self.token("1x1") or self.token("3x3")
            if c:
                if self.token("]"):
                    return c
            self.index = index

    def token(self, s):
        return self.re(r"\s*(" + re.escape(s) + r")\s*", 1)

    def string(self, s):
        if s == self.text[self.index:self.index+len(s)]:
            self.index += len(s)
            return s

    def re(self, regex, group=0):
        m = re.match(regex, self.text[self.index:])
        if m:
            self.index += len(m.group(0))
            return m.group(group)

def arch(model, x0, fpath, ext='png', is_hor=False):
    """Метод построения архитектуры сети.

        Args:
            model (torch.nn.Module): нейронная сеть.
            x0 (torch.tensor): входной вектор для сети.
            steps (int): количество итераций метода.
            fpath (str): путь к файлу для сохранения (без расширения).
            ext (str): расширение сохраняемого файла (png или pdf).
            is_hor (bool): флаг, если True, то будет использовано горизонтальное
                позиционирование рисунка, иначе вертикальное.

    """
    dot = build_graph(model, x0).build_dot()
    dot.attr('graph', rankdir='LR' if is_hor else 'TD')
    dot.format = ext
    dot.render(fpath)


class Rename():
    def __init__(self, op=None, name=None, to=None):
        assert op or name, "Either op or name must be provided"
        assert not(op and name), "Either op or name should be provided, but not both"
        assert bool(to), "The to parameter is required"
        self.to = to
        self.op = re.compile(op) if op else None
        self.name = re.compile(name) if name else None

    def apply(self, graph):
        # Copy the graph. Don't change the original.
        graph = copy.deepcopy(graph)

        for node in graph.nodes.values():
            if self.op:
                node.op = self.op.sub(self.to, node.op)
            # TODO: name is not tested yet
            if self.name:
                node.name = self.name.sub(self.to, node.name)
        return graph


FRAMEWORK_TRANSFORMS = [
    # Hide onnx: prefix
    Rename(op=r"onnx::(.*)", to=r"\1"),
    # ONNX uses Gemm for linear layers (stands for General Matrix Multiplication).
    # It's an odd name that noone recognizes. Rename it.
    Rename(op=r"Gemm", to=r"Linear"),
    # PyTorch layers that don't have an ONNX counterpart
    Rename(op=r"aten::max\_pool2d\_with\_indices", to="MaxPool"),
    # Shorten op name
    Rename(op=r"BatchNormalization", to="BatchNorm"),
]



class Fold():
    def __init__(self, pattern, op, name=None):
        # TODO: validate that op and name are valid
        self.pattern = GEParser(pattern).parse()
        self.op = op
        self.name = name

    def apply(self, graph):
        # Copy the graph. Don't change the original.
        graph = copy.deepcopy(graph)

        while True:
            matches, _ = graph.search(self.pattern)
            if not matches:
                break

            # Replace pattern with new node
            if self.op == "__first__":
                combo = matches[0]
            elif self.op == "__last__":
                combo = matches[-1]
            else:
                combo = Node(uid=graph.sequence_id(matches),
                                name=self.name or " &gt; ".join([l.title for l in matches]),
                                op=self.op or self.pattern,
                                n_train_par = sum([nod.n_train_par for nod in matches]),
                                output_shape=matches[-1].output_shape)
                combo._caption = "/".join(filter(None, [l.caption for l in matches]))
            graph.replace(matches, combo)
        return graph


class FoldId():
    def __init__(self, id_regex, op, name=None):
        # TODO: validate op and name are valid
        self.id_regex = re.compile(id_regex)
        self.op = op
        self.name = name

    def apply(self, graph):
        # Copy the graph. Don't change the original.
        graph = copy.deepcopy(graph)

        # Group nodes by the first matching group of the regex
        groups = {}
        for node in graph.nodes.values():
            m = self.id_regex.match(node.id)
            if not m:
                continue

            assert m.groups(), "Regular expression must have a matching group to avoid folding unrelated nodes."
            key = m.group(1)
            if key not in groups:
                groups[key] = []
            groups[key].append(node)

        # Fold each group of nodes together
        for key, nodes in groups.items():
            # Replace with a new node
            # TODO: Find last node in the sub-graph and get the output shape from it
            combo = Node(uid=key,
                         name=self.name,
                         op=self.op)
            graph.replace(nodes, combo)
        return graph


class Prune():
    def __init__(self, pattern):
        self.pattern = GEParser(pattern).parse()

    def apply(self, graph):
        # Copy the graph. Don't change the original.
        graph = copy.deepcopy(graph)

        while True:
            matches, _ = graph.search(self.pattern)
            if not matches:
                break
            # Remove found nodes
            graph.remove(matches)
        return graph


class PruneBranch():
    def __init__(self, pattern):
        self.pattern = GEParser(pattern).parse()

    def tag(self, node, tag, graph, conditional=False):
        # Return if the node is already tagged
        if hasattr(node, "__tag__") and node.__tag__ == "tag":
            return
        # If conditional, then tag the node if and only if all its
        # outgoing nodes already have the same tag.
        if conditional:
            # Are all outgoing nodes already tagged?
            outgoing = graph.outgoing(node)
            tagged = filter(lambda n: hasattr(n, "__tag__") and n.__tag__ == tag,
                            outgoing)
            if len(list(tagged)) != len(outgoing):
                # Not all outgoing are tagged
                return
        # Tag the node
        node.__tag__ = tag
        # Tag incoming nodes
        for n in graph.incoming(node):
            self.tag(n, tag, graph, conditional=True)

    def apply(self, graph):
        # Copy the graph. Don't change the original.
        graph = copy.deepcopy(graph)

        while True:
            matches, _ = graph.search(self.pattern)
            if not matches:
                break
            # Tag found nodes and their incoming branches
            for n in matches:
                self.tag(n, "delete", graph)
            # Find all tagged nodes and delete them
            tagged = [n for n in graph.nodes.values()
                      if hasattr(n, "__tag__") and n.__tag__ == "delete"]
            graph.remove(tagged)
        return graph


class FoldDuplicates():
    def apply(self, graph):
        # Copy the graph. Don't change the original.
        graph = copy.deepcopy(graph)

        matches = True
        while matches:
            for node in graph.nodes.values():
                pattern = SerialPattern(
                    [NodePattern(node.op), NodePattern(node.op)])
                matches, _ = pattern.match(graph, node)
                if matches:
                    # Use op and name from the first node, and output_shape from the last
                    combo = Node(uid=graph.sequence_id(matches),
                                name=node.name,
                                op=node.op,
                                output_shape=matches[-1].output_shape)
                    combo._caption = node.caption
                    combo.repeat = sum([n.repeat for n in matches])
                    combo.n_train_par = sum([n.n_train_par for n in matches])
                    graph.replace(matches, combo)
                    break
        return graph




# Transforms to simplify graphs by folding layers that tend to be
# used together often, such as Conv/BN/Relu.
# These transforms are used AFTER the framework specific transforms
# that map TF and PyTorch graphs to a common representation.
SIMPLICITY_TRANSFORMS = [
    Fold("Conv > Conv > BatchNorm > Relu", "ConvConvBnRelu"),
    Fold("Conv > BatchNorm > Relu", "ConvBnRelu"),
    Fold("Conv > BatchNorm", "ConvBn"),
    Fold("Conv > Relu", "ConvRelu"),
    Fold("Linear > Relu", "LinearRelu"),
    # Fold("ConvBnRelu > MaxPool", "ConvBnReluMaxpool"),
    # Fold("ConvRelu > MaxPool", "ConvReluMaxpool"),
    FoldDuplicates(),
]


THEMES = {
    "basic": {
        "background_color": "#FFFFFF",
        "fill_color": "#E8E8E8",
        "outline_color": "#000000",
        "font_color": "#000000",
        "font_name": "Times",
        "font_size": "10",
        "margin": "0,0",
        "padding":  "1.0,0.5",
    },
    "nice": {
        "background_color": "#FFFFFF",
        "fill_color": "#F5F5F5",
        "outline_color": "#000000",
        "font_color": "#000000",
        "font_name": "Palatino",
        "font_size": "10",
        "margin": "0,0",
        "padding":  "1.0,0.5",
    },
    "blue": {
        "background_color": "#FFFFFF",
        "fill_color": "#BCD6FC",
        "outline_color": "#7C96BC",
        "font_color": "#202020",
        "font_name": "Verdana",
        "font_size": "10",
        "margin": "0,0",
        "padding":  "1.0,0.5",
    },
}


# https://stackoverflow.com/questions/579310/formatting-long-numbers-as-strings-in-python/45846841
def human_format(num):
    num = float('{:.3g}'.format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '{}{}'.format('{:f}'.format(num).rstrip('0').rstrip('.'), ['', 'K', 'M', 'B', 'T'][magnitude])


class Node():
    def __init__(self, uid, name, op, n_train_par = 0, output_shape=None, params=None):
        """
        uid: unique ID for the layer that doesn't repeat in the computation graph.
        name: Name to display
        op: Framework-agnostic operation name.
        """
        self.id = uid
        self.name = name  # TODO: clarify the use of op vs name vs title
        self.op = op
        self.n_train_par = n_train_par
        self.repeat = 1
        if output_shape:
            assert isinstance(output_shape, (tuple, list)),\
            "output_shape must be a tuple or list but received {}".format(type(output_shape))
        self.output_shape = output_shape
        self.params = params if params else {}
        self._caption = ""

    @property
    def title(self):
        # Default
        title = self.name or self.op

        if "kernel_shape" in self.params:
            # Kernel
            kernel = self.params["kernel_shape"]
            title += "x".join(map(str, kernel))
        if "stride" in self.params:
            stride = self.params["stride"]
            if np.unique(stride).size == 1:
                stride = stride[0]
            if stride != 1:
                title += "/s{}".format(str(stride))
        #         # Transposed
        #         if node.transposed:
        #             name = "Transposed" + name
        return title

    @property
    def caption(self):
        if self._caption:
            return self._caption

        caption = ""

        # Stride
        # if "stride" in self.params:
        #     stride = self.params["stride"]
        #     if np.unique(stride).size == 1:
        #         stride = stride[0]
        #     if stride != 1:
        #         caption += "/{}".format(str(stride))
        return caption

    def __repr__(self):
        args = (self.op, self.name, self.id, self.title, self.repeat)
        f = "<Node: op: {}, name: {}, id: {}, title: {}, repeat: {}"
        if self.output_shape:
            args += (str(self.output_shape),)
            f += ", shape: {:}"
        if self.params:
            args += (str(self.params),)
            f += ", params: {:}"
        f += ">"
        return f.format(*args)


def build_graph(model=None, args=None, input_names=None,
                transforms="default", framework_transforms="default"):

    g = Graph()

    import_graph(g, model, args)

    if framework_transforms:
        if framework_transforms == "default":
            framework_transforms = FRAMEWORK_TRANSFORMS
        for t in framework_transforms:
            g = t.apply(g)
    if transforms:
        if transforms == "default":
            transforms = SIMPLICITY_TRANSFORMS
        for t in transforms:
            g = t.apply(g)
    return g


class Graph():
    """Tracks nodes and edges of a directed graph and supports basic operations on them."""

    def __init__(self, model=None, args=None, input_names=None,
                 transforms="default", framework_transforms="default",
                 meaningful_ids=False):
        self.nodes = {}
        self.edges = []
        self.meaningful_ids = meaningful_ids # TODO
        self.theme = THEMES["nice"]

        if model:
            import_graph(self, model, args)

            if framework_transforms:
                if framework_transforms == "default":
                    framework_transforms = FRAMEWORK_TRANSFORMS
                for t in framework_transforms:
                    t.apply(self)
            if transforms:
                if transforms == "default":
                    transforms = SIMPLICITY_TRANSFORMS
                for t in transforms:
                    t.apply(self)


    def id(self, node):
        """Returns a unique node identifier. If the node has an id
        attribute (preferred), it's used. Otherwise, the hash() is returned."""
        return node.id if hasattr(node, "id") else hash(node)

    def add_node(self, node):
        id = self.id(node)
        # assert(id not in self.nodes)
        self.nodes[id] = node

    def add_edge(self, node1, node2, label=None):
        # If the edge is already present, don't add it again.
        # TODO: If an edge exists with a different label, still don't add it again.
        edge = (self.id(node1), self.id(node2), label)
        if edge not in self.edges:
            self.edges.append(edge)

    def add_edge_by_id(self, vid1, vid2, label=None):
        self.edges.append((vid1, vid2, label))

    def outgoing(self, node):
        """Returns nodes connecting out of the given node (or list of nodes)."""
        nodes = node if isinstance(node, list) else [node]
        node_ids = [self.id(n) for n in nodes]
        # Find edges outgoing from this group but not incoming to it
        outgoing = [self[e[1]] for e in self.edges
                    if e[0] in node_ids and e[1] not in node_ids]
        return outgoing

    def incoming(self, node):
        """Returns nodes connecting to the given node (or list of nodes)."""
        nodes = node if isinstance(node, list) else [node]
        node_ids = [self.id(n) for n in nodes]
        # Find edges incoming to this group but not outgoing from it
        incoming = [self[e[0]] for e in self.edges
                    if e[1] in node_ids and e[0] not in node_ids]
        return incoming

    def siblings(self, node):
        """Returns all nodes that share the same parent (incoming node) with
        the given node, including the node itself.
        """
        incoming = self.incoming(node)
        # TODO: Not handling the case of multiple incoming nodes yet
        if len(incoming) == 1:
            incoming = incoming[0]
            siblings = self.outgoing(incoming)
            return siblings
        else:
            return [node]

    def __getitem__(self, key):
        if isinstance(key, list):
            return [self.nodes.get(k) for k in key]
        else:
            return self.nodes.get(key)

    def remove(self, nodes):
        """Remove a node and its edges."""
        nodes = nodes if isinstance(nodes, list) else [nodes]
        for node in nodes:
            k = self.id(node)
            self.edges = list(filter(lambda e: e[0] != k and e[1] != k, self.edges))
            del self.nodes[k]

    def replace(self, nodes, node):
        """Replace nodes with node. Edges incoming to nodes[0] are connected to
        the new node, and nodes outgoing from nodes[-1] become outgoing from
        the new node."""
        nodes = nodes if isinstance(nodes, list) else [nodes]
        # Is the new node part of the replace nodes (i.e. want to collapse
        # a group of nodes into one of them)?
        collapse = self.id(node) in self.nodes
        # Add new node and edges
        if not collapse:
            self.add_node(node)
        for in_node in self.incoming(nodes):
            # TODO: check specifically for output_shape is not generic. Consider refactoring.
            self.add_edge(in_node, node, in_node.output_shape if hasattr(in_node, "output_shape") else None)
        for out_node in self.outgoing(nodes):
            self.add_edge(node, out_node, node.output_shape if hasattr(node, "output_shape") else None)
        # Remove the old nodes
        for n in nodes:
            if collapse and n == node:
                continue
            self.remove(n)

    def search(self, pattern):
        """Searches the graph for a sub-graph that matches the given pattern
        and returns the first match it finds.
        """
        for node in self.nodes.values():
            match, following = pattern.match(self, node)
            if match:
                return match, following
        return [], None


    def sequence_id(self, sequence):
        """Make up an ID for a sequence (list) of nodes.
        Note: `getrandbits()` is very uninformative as a "readable" ID. Here, we build a name
        such that when the mouse hovers over the drawn node in Jupyter, one can figure out
        which original nodes make up the sequence. This is actually quite useful.
        """
        if self.meaningful_ids:
            # TODO: This might fail if the ID becomes too long
            return "><".join([node.id for node in sequence])
        else:
            return getrandbits(64)

    def build_dot(self):
        """Generate a GraphViz Dot graph.

        Returns a GraphViz Digraph object.
        """
        from graphviz import Digraph

        # Build GraphViz Digraph
        dot = Digraph()
        dot.attr("graph",
                 bgcolor=self.theme["background_color"],
                 color=self.theme["outline_color"],
                 fontsize=self.theme["font_size"],
                 fontcolor=self.theme["font_color"],
                 fontname=self.theme["font_name"],
                 margin=self.theme["margin"],
                 rankdir="TD",
                 pad=self.theme["padding"])
        dot.attr("node", shape="box",
                 style="filled, rounded", margin="0.05,0.05",
                 fillcolor=self.theme["fill_color"],
                 color=self.theme["outline_color"],
                 fontsize=self.theme["font_size"],
                 fontcolor=self.theme["font_color"],
                 fontname=self.theme["font_name"])
        dot.attr("edge", style="solid",
                 color=self.theme["outline_color"],
                 fontsize=self.theme["font_size"],
                 fontcolor=self.theme["font_color"],
                 fontname=self.theme["font_name"])

        for k, n in self.nodes.items():
            label = "<tr><td cellpadding='2'>{}</td></tr>".format(n.title.replace('&gt;','+'))
            if n.caption:
                label += "<tr><td>{}</td></tr>".format(n.caption)
            if n.n_train_par > 0:
                label += f"<tr><td>{human_format(n.n_train_par)} parameters</td></tr>"
            if n.repeat > 1:
                label += "<tr><td align='right'>x{}</td></tr>".format(n.repeat)
            label = "<<table border='0' cellborder='0' cellpadding='0'>" + label + "</table>>"
            dot.node(str(k), label)
        for a, b, label in self.edges:
            if isinstance(label, (list, tuple)):
                label = "x".join([str(l or "?") for l in label])

            dot.edge(str(a), str(b), label)
        return dot

    def _repr_svg_(self):
        """Allows Jupyter notebook to render the graph automatically."""
        return self.build_dot()._repr_svg_()

    def save(self, path, format="pdf"):
        # TODO: assert on acceptable format values
        dot = self.build_dot()
        dot.format = format
        directory, file_name = os.path.split(path)
        # Remove extension from file name. dot.render() adds it.
        file_name = file_name.replace("." + format, "")
        dot.render(file_name, directory=directory, cleanup=True)



def dump_pytorch_graph(graph):
    """List all the nodes in a PyTorch graph."""
    f = "{:25} {:40}   {} -> {}"
    print(f.format("kind", "scopeName", "inputs", "outputs"))
    for node in graph.nodes():
        print(f.format(node.kind(), node.scopeName(),
                       [i.unique() for i in node.inputs()],
                       [i.unique() for i in node.outputs()]
                       ))


def pytorch_id(node):
    """Returns a unique ID for a node."""
    # After ONNX simplification, the scopeName is not unique anymore
    # so append node outputs to guarantee uniqueness
    return node.scopeName() + "/outputs/" + "/".join(["{}".format(o.unique()) for o in node.outputs()])


def get_shape(torch_node):
    """Return the output shape of the given Pytorch node."""
    # Extract node output shape from the node string representation
    # This is a hack because there doesn't seem to be an official way to do it.
    # See my quesiton in the PyTorch forum:
    # https://discuss.pytorch.org/t/node-output-shape-from-trace-graph/24351/2
    # TODO: find a better way to extract output shape
    # TODO: Assuming the node has one output. Update if we encounter a multi-output node.
    m = re.match(r".*Float\(([\d\s\,]+)\).*", str(next(torch_node.outputs())))
    if m:
        shape = m.group(1)
        shape = shape.split(",")
        shape = tuple(map(int, shape))
    else:
        shape = None
    return shape


def import_graph(hl_graph, model, args, input_names=None, verbose=False):
    # TODO: add input names to graph
    # For simplicity we consider only layers with trainable parameters
    onnx_to_torch_summary_dict = {
        "onnx::Conv": "Conv",
        "onnx::BatchNormalization": "BatchNorm",
        # "onnx::Relu": "ReLU",
        # "onnx::MaxPool": "MaxPool",
        # "onnx::GlobalAveragePool": "AdaptiveAvgPool",
        "onnx::Gemm": "Linear"
    }

    # Run the Pytorch graph to get a trace and generate a graph from it
    trace, out = torch.jit._get_trace_graph(model, args)
    torch_graph = torch.onnx._optimize_trace(trace, torch.onnx.OperatorExportTypes.ONNX)

    torch_summ_dict = torch_model_dict(model, input_size=tuple(args.shape[1:]))

    torch_summ_dict_copy = copy.deepcopy(torch_summ_dict)

    # Dump list of nodes (DEBUG only)
    if verbose:
        dump_pytorch_graph(torch_graph)

    # Loop through nodes and build HL graph
    for torch_node in torch_graph.nodes():
        # Op
        op = torch_node.kind()
        # Number of training parameters for each layer
        n_train_par = 0
        if op in onnx_to_torch_summary_dict.keys():
            for key in torch_summ_dict_copy.keys():
                if onnx_to_torch_summary_dict[op] in key:
                    n_train_par = int(torch_summ_dict_copy[key]['nb_params'])
                    del torch_summ_dict_copy[key]
                    break
        # Parameters
        params = {k: torch_node[k] for k in torch_node.attributeNames()}
        # Inputs/outputs
        # TODO: inputs = [i.unique() for i in node.inputs()]
        outputs = [o.unique() for o in torch_node.outputs()]
        # Get output shape
        shape = get_shape(torch_node)
        # Add HL node
        hl_node = Node(uid=pytorch_id(torch_node), name=None, op=op, n_train_par=n_train_par,
                       output_shape=shape, params=params)
        hl_graph.add_node(hl_node)
        # Add edges
        for target_torch_node in torch_graph.nodes():
            target_inputs = [i.unique() for i in target_torch_node.inputs()]
            if set(outputs) & set(target_inputs):
                hl_graph.add_edge_by_id(pytorch_id(torch_node), pytorch_id(target_torch_node), shape)
    return hl_graph



def torch_model_dict(model, input_size):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    def summary_string(model, input_size, batch_size=-1, device=device, dtypes=None, return_dict = False):
        if dtypes == None:
            dtypes = [torch.FloatTensor]*len(input_size)

        summary_str = ''

        def register_hook(module):
            def hook(module, input, output):
                class_name = str(module.__class__).split(".")[-1].split("'")[0]
                module_idx = len(summary)

                m_key = "%s-%i" % (class_name, module_idx + 1)
                summary[m_key] = OrderedDict()
                summary[m_key]["input_shape"] = list(input[0].size())
                summary[m_key]["input_shape"][0] = batch_size
                if isinstance(output, (list, tuple)):
                    summary[m_key]["output_shape"] = [
                        [-1] + list(o.size())[1:] for o in output
                    ]
                else:
                    summary[m_key]["output_shape"] = list(output.size())
                    summary[m_key]["output_shape"][0] = batch_size

                params = 0
                if hasattr(module, "weight") and hasattr(module.weight, "size"):
                    params += torch.prod(torch.LongTensor(list(module.weight.size())))
                    summary[m_key]["trainable"] = module.weight.requires_grad
                if hasattr(module, "bias") and hasattr(module.bias, "size"):
                    params += torch.prod(torch.LongTensor(list(module.bias.size())))
                summary[m_key]["nb_params"] = params

            if (
                not isinstance(module, nn.Sequential)
                and not isinstance(module, nn.ModuleList)
            ):
                hooks.append(module.register_forward_hook(hook))

        # multiple inputs to the network
        if isinstance(input_size, tuple):
            input_size = [input_size]

        # batch_size of 2 for batchnorm
        x = [torch.rand(2, *in_size).type(dtype).to(device=device)
             for in_size, dtype in zip(input_size, dtypes)]

        # create properties
        summary = OrderedDict()
        hooks = []

        # register hook
        model.apply(register_hook)

        # make a forward pass
        # print(x.shape)
        model(*x)

        # remove these hooks
        for h in hooks:
            h.remove()

        summary_str += "----------------------------------------------------------------" + "\n"
        line_new = "{:>20}  {:>25} {:>15}".format(
            "Layer (type)", "Output Shape", "Param #")
        summary_str += line_new + "\n"
        summary_str += "================================================================" + "\n"
        total_params = 0
        total_output = 0
        trainable_params = 0
        for layer in summary:
            # input_shape, output_shape, trainable, nb_params
            line_new = "{:>20}  {:>25} {:>15}".format(
                layer,
                str(summary[layer]["output_shape"]),
                "{0:,}".format(summary[layer]["nb_params"]),
            )
            total_params += summary[layer]["nb_params"]

            total_output += np.prod(summary[layer]["output_shape"])
            if "trainable" in summary[layer]:
                if summary[layer]["trainable"] == True:
                    trainable_params += summary[layer]["nb_params"]
            summary_str += line_new + "\n"

        # assume 4 bytes/number (float on cuda).
        total_input_size = abs(np.prod(sum(input_size, ()))
                               * batch_size * 4. / (1024 ** 2.))
        total_output_size = abs(2. * total_output * 4. /
                                (1024 ** 2.))  # x2 for gradients
        total_params_size = abs(total_params * 4. / (1024 ** 2.))
        total_size = total_params_size + total_output_size + total_input_size

        summary_str += "================================================================" + "\n"
        summary_str += "Total params: {0:,}".format(total_params) + "\n"
        summary_str += "Trainable params: {0:,}".format(trainable_params) + "\n"
        summary_str += "Non-trainable params: {0:,}".format(total_params -
                                                            trainable_params) + "\n"
        summary_str += "----------------------------------------------------------------" + "\n"
        summary_str += "Input size (MB): %0.2f" % total_input_size + "\n"
        summary_str += "Forward/backward pass size (MB): %0.2f" % total_output_size + "\n"
        summary_str += "Params size (MB): %0.2f" % total_params_size + "\n"
        summary_str += "Estimated Total Size (MB): %0.2f" % total_size + "\n"
        summary_str += "----------------------------------------------------------------" + "\n"
        if return_dict:
            return summary
        else:
            return summary_str, (total_params, trainable_params)

    return summary_string(model, input_size, device = device, return_dict = True)
