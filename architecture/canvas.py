from collections import OrderedDict
import copy
from graphviz import Digraph
import numpy as np
import os
from random import getrandbits
import re
import torch
import torch.nn as nn


class Graph:
    def __init__(self):
        self.nodes = {}
        self.edges = []
        self.theme = {
            "background_color": "#FFFFFF",
            "fill_color": "#F5F5F5",
            "outline_color": "#000000",
            "font_color": "#000000",
            "font_name": "Palatino",
            "font_size": "10",
            "margin": "0,0",
            "padding":  "1.0,0.5",
        }

    def create(self, model, args):
        onnx_to_torch_summary_dict = {
            "onnx::Conv": "Conv",
            "onnx::BatchNormalization": "BatchNorm",
            "onnx::Gemm": "Linear"
        }

        def get_shape(torch_node):
            m = re.match(r".*Float\(([\d\s\,]+)\).*", str(next(torch_node.outputs())))
            if m:
                shape = m.group(1)
                shape = shape.split(",")
                shape = tuple(map(int, shape))
            else:
                shape = None
            return shape
        def pytorch_id(node):
            return node.scopeName() + "/outputs/" + "/".join(["{}".format(o.unique()) for o in node.outputs()])

        trace, out = torch.jit._get_trace_graph(model, args)
        torch_graph = torch.onnx._optimize_trace(trace, torch.onnx.OperatorExportTypes.ONNX)

        torch_summ_dict = model_info(model, input_size=tuple(args.shape[1:]))

        torch_summ_dict_copy = copy.deepcopy(torch_summ_dict)

        for torch_node in torch_graph.nodes():
            op = torch_node.kind()
            n_train_par = 0
            if op in onnx_to_torch_summary_dict.keys():
                for key in torch_summ_dict_copy.keys():
                    if onnx_to_torch_summary_dict[op] in key:
                        n_train_par = int(torch_summ_dict_copy[key]['nb_params'])
                        del torch_summ_dict_copy[key]
                        break
            params = {k: torch_node[k] for k in torch_node.attributeNames()}
            outputs = [o.unique() for o in torch_node.outputs()]
            shape = get_shape(torch_node)
            hl_node = Node(uid=pytorch_id(torch_node), name=None, op=op, n_train_par=n_train_par,
                           output_shape=shape, params=params)
            self.add_node(hl_node)
            for target_torch_node in torch_graph.nodes():
                target_inputs = [i.unique() for i in target_torch_node.inputs()]
                if set(outputs) & set(target_inputs):
                    self.add_edge_by_id(pytorch_id(torch_node), pytorch_id(target_torch_node), shape)
        return self

    def id(self, node):
        return node.id if hasattr(node, "id") else hash(node)

    def add_node(self, node):
        id = self.id(node)
        self.nodes[id] = node

    def add_edge(self, node1, node2, label=None):
        edge = (self.id(node1), self.id(node2), label)
        if edge not in self.edges:
            self.edges.append(edge)

    def add_edge_by_id(self, vid1, vid2, label=None):
        self.edges.append((vid1, vid2, label))

    def outgoing(self, node):
        nodes = node if isinstance(node, list) else [node]
        node_ids = [self.id(n) for n in nodes]
        outgoing = [self[e[1]] for e in self.edges
                    if e[0] in node_ids and e[1] not in node_ids]
        return outgoing

    def incoming(self, node):
        nodes = node if isinstance(node, list) else [node]
        node_ids = [self.id(n) for n in nodes]
        incoming = [self[e[0]] for e in self.edges
                    if e[1] in node_ids and e[0] not in node_ids]
        return incoming

    def siblings(self, node):
        incoming = self.incoming(node)
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
        nodes = nodes if isinstance(nodes, list) else [nodes]
        for node in nodes:
            k = self.id(node)
            self.edges = list(filter(lambda e: e[0] != k and e[1] != k, self.edges))
            del self.nodes[k]

    def replace(self, nodes, node):
        nodes = nodes if isinstance(nodes, list) else [nodes]
        collapse = self.id(node) in self.nodes
        if not collapse:
            self.add_node(node)
        for in_node in self.incoming(nodes):
            self.add_edge(in_node, node, in_node.output_shape if hasattr(in_node, "output_shape") else None)
        for out_node in self.outgoing(nodes):
            self.add_edge(node, out_node, node.output_shape if hasattr(node, "output_shape") else None)
        for n in nodes:
            if collapse and n == node:
                continue
            self.remove(n)

    def search(self, pattern):
        for node in self.nodes.values():
            match, following = pattern.match(self, node)
            if match:
                return match, following
        return [], None


    def sequence_id(self, sequence):
        return getrandbits(64)

    def build_dot(self):
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

        def human_format(num):
            num = float('{:.3g}'.format(num))
            magnitude = 0
            while abs(num) >= 1000:
                magnitude += 1
                num /= 1000.0
            return '{}{}'.format('{:f}'.format(num).rstrip('0').rstrip('.'), ['', 'K', 'M', 'B', 'T'][magnitude])

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
        return self.build_dot()._repr_svg_()

    def save(self, path, format="pdf"):
        dot = self.build_dot()
        dot.format = format
        directory, file_name = os.path.split(path)
        file_name = file_name.replace("." + format, "")
        dot.render(file_name, directory=directory, cleanup=True)


class Node:
    def __init__(self, uid, name, op, n_train_par = 0, output_shape=None, params=None):
        self.id = uid
        self.name = name
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
        return title

    @property
    def caption(self):
        if self._caption:
            return self._caption

        caption = ""
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


class NodePattern:
    def __init__(self, op):
        self.op = op

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


class Parser:
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
            return PatternParallel(expressions)
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
            return PatternSerial(expressions)
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
            return NodePattern(t)

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


class PatternSerial:
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
                node = following
        return all_matches, following


class PatternParallel:
    def __init__(self, patterns):
        self.patterns = patterns

    def match(self, graph, nodes):
        if not nodes:
            return [], None
        nodes = nodes if isinstance(nodes, list) else [nodes]
        if len(nodes) == 1:
            nodes = graph.siblings(nodes[0])
        else:
            parents = [graph.incoming(n) for n in nodes]
            matches = [set(p) == set(parents[0]) for p in parents[1:]]
            if not all(matches):
                return [], None
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
                    if end_node:
                        if end_node != following:
                            return [], None
                    else:
                        end_node = following
                    break
            if not found:
                return [], None
        return all_matches, end_node


class TransFold:
    def __init__(self, pattern, op, name=None):
        self.pattern = Parser(pattern).parse()
        self.op = op
        self.name = name

    def apply(self, graph):
        graph = copy.deepcopy(graph)

        while True:
            matches, _ = graph.search(self.pattern)
            if not matches:
                break

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


class TransFoldDuplicates:
    def apply(self, graph):
        graph = copy.deepcopy(graph)

        matches = True
        while matches:
            for node in graph.nodes.values():
                pattern = PatternSerial(
                    [NodePattern(node.op), NodePattern(node.op)])
                matches, _ = pattern.match(graph, node)
                if matches:
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


class TransRename:
    def __init__(self, op=None, name=None, to=None):
        assert op or name, "Either op or name must be provided"
        assert not(op and name), "Either op or name should be provided, but not both"
        assert bool(to), "The to parameter is required"
        self.to = to
        self.op = re.compile(op) if op else None
        self.name = re.compile(name) if name else None

    def apply(self, graph):
        graph = copy.deepcopy(graph)

        for node in graph.nodes.values():
            if self.op:
                node.op = self.op.sub(self.to, node.op)
            if self.name:
                node.name = self.name.sub(self.to, node.name)
        return graph


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
    dot = build(model, x0).build_dot()
    dot.attr('graph', rankdir='LR' if is_hor else 'TD')
    dot.format = ext
    dot.render(fpath)


def build(model=None, args=None, input_names=None, transforms="default"):
    transforms = [
        TransFold("Conv > Conv > BatchNorm > Relu", "ConvConvBnRelu"),
        TransFold("Conv > BatchNorm > Relu", "ConvBnRelu"),
        TransFold("Conv > BatchNorm", "ConvBn"),
        TransFold("Conv > Relu", "ConvRelu"),
        TransFold("Linear > Relu", "LinearRelu"),
        TransFoldDuplicates(),
    ]

    framework_transforms = [
        TransRename(op=r"onnx::(.*)", to=r"\1"),
        TransRename(op=r"Gemm", to=r"Linear"),
        TransRename(op=r"aten::max\_pool2d\_with\_indices", to="MaxPool"),
        TransRename(op=r"BatchNormalization", to="BatchNorm"),
    ]

    g = Graph().create(model, args)
    for t in framework_transforms:
        g = t.apply(g)
    for t in transforms:
        g = t.apply(g)
    return g


def model_info(model, input_size):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_size = -1
    device = device
    dtypes = [torch.FloatTensor]*len(input_size)

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

    if isinstance(input_size, tuple):
        input_size = [input_size]

    x = [torch.rand(2, *in_size).type(dtype).to(device=device)
         for in_size, dtype in zip(input_size, dtypes)]

    summary = OrderedDict()
    hooks = []

    model.apply(register_hook)

    model(*x)
    for h in hooks:
        h.remove()

    return summary
