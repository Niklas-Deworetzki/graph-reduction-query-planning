import json
from datetime import datetime
from pathlib import Path
from typing import Iterator

from flask import Flask, abort, render_template

from grqe.profiling import to_graphviz

server = Flask(__name__, template_folder='static')

TRACES_PATH = Path(__file__).parent.parent / 'traces'


def get_traces() -> Iterator[str]:
    for filename in TRACES_PATH.iterdir():
        if filename.suffix == '.jsonl':
            with open(filename) as trace_file:
                yield from trace_file


@server.get("/")
def main():
    traces = map(json.loads, get_traces())
    return _list_traces(enumerate(traces))


@server.get('/search/<query>')
def get_searched(query: str):
    return _list_traces(_search(query))

@server.get('/traces/<n>')
def get_trace(n: int):
    n = int(n)
    for i, trace in enumerate(get_traces()):
        if i == n:
            svg = to_graphviz(trace).pipe(format='svg')
            return _render(svg)
    abort(404)


def _render(svg: bytes) -> str:
    return render_template('svg_viewport.html', svg=svg.decode())


def _list_traces(traces: Iterator[tuple[int, dict]]):
    queries = [
        (
            data.get('original', 'MISSING DATA'),
            datetime.fromtimestamp(data['time']).strftime('%Y-%m-%d %H:%M:%S') if 'time' in data else 'MISSING_DATA',
            n
        )
        for n, data in traces
    ]
    return render_template('query_table.html', queries=queries)


def _search(query: str) -> Iterator[tuple[int, dict]]:
    for i, trace in enumerate(get_traces()):
        data = json.loads(trace)
        if query in data.get('original', set()):
            yield i, data


server.run('localhost', 5000)
