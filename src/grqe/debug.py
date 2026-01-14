import logging
import time
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from typing import Callable, ClassVar, Optional

import graphviz

from grqe.query import Lookup, Node

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


_GRAPHVIZ_NODE_STYLE = {
    'shape': 'box',
    'style': 'filled',
}


def visualize(
        root: Node,
        score: Callable[[Node], int],
        score_format: Callable[[int], str] = str,
        comment: str = None
) -> graphviz.Digraph:
    all_nodes = list(root.flatten())
    node_ids = {
        id(node): str(i)
        for i, node in enumerate(root.flatten())
    }
    min_score = min(score(n) for n in all_nodes)
    max_score = max(score(n) for n in all_nodes)

    def display(n: Node) -> str:
        if isinstance(n, Lookup):
            atoms = [f'{atom.key}@{atom.relative_position}=\"{atom.value}\"' for atom in n.atoms]
            return '[' + ', '.join(atoms) + ']'
        else:
            return str(type(n).__name__)

    def interpolate_color(col_min: str, col_max: str, value: int) -> str:
        res = '#'
        for channel in range(3):
            chan_min = int(col_min[1 + 2 * channel: 1 + 2 * (channel + 1)], 16)
            chan_max = int(col_max[1 + 2 * channel: 1 + 2 * (channel + 1)], 16)

            chan_val = (value - min_score) / (max_score - min_score) * chan_max + \
                       (1 - (value - min_score) / (max_score - min_score)) * chan_min
            chan_val = round(chan_val)
            res += f'{chan_val:0>2X}'
        return res

    graph = graphviz.Digraph(node_attr=_GRAPHVIZ_NODE_STYLE, comment=comment or 'auto-generated graphviz')
    for node in all_nodes:
        node_text = f'{display(node)}\\n{score_format(score(node))}\\n'
        node_color = interpolate_color('#FBEF76', '#FA5C5C', score(node))
        str_id = node_ids[id(node)]
        graph.node(str_id, node_text, fillcolor=node_color)

    def edges(n: Node, inbound: Optional[str]):
        str_id = node_ids[id(n)]
        if inbound:
            graph.edge(inbound, str_id)

        for child in n.children():
            edges(child, str_id)

    edges(root, None)
    return graph
