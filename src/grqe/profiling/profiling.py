import time
from collections import defaultdict
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any, Callable, ClassVar

from grqe.query import Lookup, Node, SpanLookup

current_time: Callable[[], int] = time.perf_counter_ns


def _node_representation(node: Node) -> str:
    match node:
        case Lookup():
            atoms = [f'{atom.key}@{atom.relative_position}=\"{atom.value}\"' for atom in node.atoms]
            return '[' + ', '.join(atoms) + ']'
        case SpanLookup():
            atoms = [f'{atom.key}=\"{atom.value}\"' for atom in node.atoms]
            if atoms:
                return f'<{node.span} {', '.join(atoms)}>'
            return f'<{node.span}>'
    return str(type(node).__name__)


def extract_profiling_trace(query: Node, original_query: str = None) -> dict:
    node_ids = {
        id(node): str(i)
        for i, node in enumerate(query.flatten(), start=1)
    }

    edges = defaultdict(list)
    profiling_info = {}
    representations = {}

    def collect(node: Node):
        node_id = node_ids[id(node)]

        profiling_info[node_id] = node._profiling_info or {}
        representations[node_id] = _node_representation(node)
        for child in node.children():
            edges[node_ids[id(child)]].append(node_id)
            collect(child)

    collect(query)

    return {
        'original': original_query,
        'time': time.time(),
        'root': node_ids[id(query)],
        'nodes': representations,
        'edges': edges,
        'profiling_info': profiling_info,
        'meta_profiling_info': commit_profiling_data(),
    }


PROFILING_ENABLED = True

if PROFILING_ENABLED:

    @dataclass
    class ProfilingMeasurement(AbstractContextManager):
        task: str
        metadata: dict[str, Any]
        time_taken: int = 0

        MEASUREMENTS_TAKEN: ClassVar[list[ProfilingMeasurement]] = []

        def __enter__(self):
            self.time_taken = current_time()
            return self

        def __exit__(self, exc_type, exc_value, traceback, /):
            self.time_taken = current_time() - self.time_taken
            ProfilingMeasurement.MEASUREMENTS_TAKEN.append(self)

        def __setitem__(self, key: str, value: Any) -> None:
            self.metadata[key] = value


    def profile(task: str, **kwargs):
        return ProfilingMeasurement(task, {key: str(value) for key, value in kwargs.items()})

    def commit_profiling_data(**additional_data) -> dict[str, Any]:
        data = {}
        for measurement in ProfilingMeasurement.MEASUREMENTS_TAKEN:
            data[measurement.task] = measurement.time_taken
        data.update(additional_data)

        ProfilingMeasurement.MEASUREMENTS_TAKEN.clear()
        return data

else:

    class ProfilingMeasurement(AbstractContextManager):
        INSTANCE: ClassVar

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback, /):
            pass

        def __setitem__(self, key: str, value: Any) -> None:
            pass


    ProfilingMeasurement.INSTANCE = ProfilingMeasurement()


    def profile(task: str):
        return ProfilingMeasurement.INSTANCE


    def commit_profiling_data(**additional_data) -> dict[str, Any]:
        return {}
