import logging
import time
from contextlib import AbstractContextManager
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


class Stopwatch(AbstractContextManager):
    task: str

    no_runs: int = 0
    accumulated_runtime: int = 0

    WATCHES: ClassVar[dict[str, Stopwatch]] = {}
    WATCH_STACK: ClassVar[list[Stopwatch]] = []

    def __init__(self, task: str):
        self.task = task
        self.no_runs = 0
        self.accumulated_runtime = 0

    def __enter__(self):
        self._start = current_time()
        Stopwatch.WATCH_STACK.append(self)
        return self

    def __exit__(self, *exc):
        assert self == Stopwatch.WATCH_STACK.pop()

        elapsed = current_time() - self._start
        LOGGER.debug(' ' * len(Stopwatch.WATCH_STACK) + f'Finished {self.task} in {(elapsed // 1000) / 1000} ms')

        self.accumulated_runtime += elapsed
        self.no_runs += 1

    @classmethod
    def for_name(cls, name: str) -> Stopwatch:
        watch = Stopwatch.WATCHES.get(name)
        if watch is None:
            watch = Stopwatch(name)
            Stopwatch.WATCHES[name] = watch
        return watch


stopwatch = Stopwatch.for_name

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
    all_nodes = {
        node: str(i)
        for i, node in enumerate(root.flatten())
    }
    min_score = min(score(n) for n in all_nodes.keys())
    max_score = max(score(n) for n in all_nodes.keys())

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
    for node, str_id in all_nodes.items():
        node_text = f'{display(node)}\\n{score_format(score(node))}\\n'
        node_color = interpolate_color('#FBEF76', '#FA5C5C', score(node))
        graph.node(str_id, node_text, fillcolor=node_color)

    def edges(n: Node, inbound: Optional[str]):
        str_id = all_nodes[n]
        if inbound:
            graph.edge(inbound, str_id)

        for child in n.children():
            edges(child, str_id)

    edges(root, None)
    return graph
