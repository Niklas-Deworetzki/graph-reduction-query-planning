import logging
import time
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from typing import Any, Callable, ClassVar, Optional, Tuple

import graphviz

from grqe.query import Lookup, Node, SpanLookup

current_time: Callable[[], int] = time.perf_counter_ns


def make_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt="[%(asctime)s] %(levelname)s: %(message)s",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    return logger


LOGGER = make_logger(__name__)

PROFILING_ENABLED = False


@dataclass
class ProfilingMeasurement(AbstractContextManager):
    task: str
    metadata: dict = field(default_factory=dict)
    time_taken: int = 0

    MEASUREMENTS_TAKEN: ClassVar[list[ProfilingMeasurement]] = []

    def __enter__(self):
        self.time_taken = current_time()
        return self

    def __exit__(self, exc_type, exc_value, traceback, /):
        self.time_taken = current_time() - self.time_taken
        if PROFILING_ENABLED:
            ProfilingMeasurement.MEASUREMENTS_TAKEN.append(self)


def profile(task: str, **kwargs):
    return ProfilingMeasurement(task, metadata=kwargs)


def format_time(ns: int) -> str:
    if ns < 1000:
        return str(ns)
    if ns < 1000 * 1000:
        return '%.2fµs' % (ns / 1000)
    if ns < 1000 * 1000 * 1000:
        return '%.2fms' % (ns / 1000 / 1000)
    return '%.2fs' % (ns / 1000 / 1000 / 1000)


_GRAPHVIZ_NODE_STYLE = {
    'shape': 'box',
    'style': 'filled',
}

_GRAPHVIZ_EDGE_STYLE = {
    'dir': 'back',
}

_GRAPHVIZ_STYLE = {
    'fontname': 'monospace',
    'fontsize': '8',
}

MIN_COLOR = '#FBEF76'
MAX_COLOR = '#FA5C5C'

MIN_EDGE_WEIGHT = 0.5
MAX_EDGE_WEIGHT = 3


def _graphviz_node_display(node: Node) -> str:
    match node:
        case Lookup():
            atoms = [f'{atom.key}@{atom.relative_position}=\"{atom.value}\"' for atom in node.atoms]
            return '[' + ', '.join(atoms) + ']'
        case SpanLookup():
            atoms = [f'{atom.key}=\"{atom.value}\"' for atom in node.atoms]
            return f'<{node.span} {', '.join(atoms)}>'
    return str(type(node).__name__)


def visualize_execution_stats(root: Node, comment: str = None) -> graphviz.Digraph:
    all_nodes = list(root.flatten())

    max_execution_time = max(n.time for n in all_nodes)
    min_execution_time = min(n.time for n in all_nodes)

    max_result_size = max(len(n.value) for n in all_nodes)
    min_result_size = min(len(n.value) for n in all_nodes)

    node_ids = {
        id(node): str(i)
        for i, node in enumerate(all_nodes, start=1)
    }

    def interpolate_color(col_min: str, col_max: str, val_min: int, val_max: int, value: int) -> str:
        if val_min == val_max:
            return col_max

        res = '#'
        for channel in range(3):
            chan_min = int(col_min[1 + 2 * channel: 1 + 2 * (channel + 1)], 16)
            chan_max = int(col_max[1 + 2 * channel: 1 + 2 * (channel + 1)], 16)

            perc = (value - val_min) / (val_max - val_min)
            chan_val = perc * chan_max + (1 - perc) * chan_min
            chan_val = round(chan_val)
            res += f'{chan_val:0>2X}'
        return res

    graph = graphviz.Digraph(
        comment=comment or 'auto-generated graphviz',
        node_attr=_GRAPHVIZ_NODE_STYLE | _GRAPHVIZ_STYLE,
        edge_attr=_GRAPHVIZ_EDGE_STYLE | _GRAPHVIZ_STYLE,
    )
    graph.node('0', style='invis')

    for node in all_nodes:
        node_text = f'{_graphviz_node_display(node)}\\n{format_time(node.time)}\\n'
        node_color = interpolate_color(
            MIN_COLOR, MAX_COLOR,
            min_execution_time, max_execution_time,
            node.time)

        str_id = node_ids[id(node)]
        graph.node(str_id, node_text, fillcolor=node_color)

    def edges(n: Node, inbound: Optional[str]):
        str_id = node_ids[id(n)]
        if inbound:
            edge_text = str(len(n.value))
            edge_color = interpolate_color(
                MIN_COLOR, MAX_COLOR,
                min_result_size, max_result_size,
                len(n.value)
            )

            perc = (len(n.value) - min_result_size) / (
                        max_result_size - min_result_size) if max_result_size != min_result_size else 1
            edge_weight = perc * MAX_EDGE_WEIGHT + (1 - perc) * MIN_EDGE_WEIGHT
            graph.edge(inbound, str_id, edge_text, fillcolor=edge_color, arrowsize=str(edge_weight))

        for child in n.children():
            edges(child, str_id)

    edges(root, '0')
    return graph
