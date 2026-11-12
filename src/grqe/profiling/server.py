import json
from datetime import datetime
from pathlib import Path
from typing import Any, Generator

from flask import Flask, abort, render_template

from grqe.profiling import to_graphviz


class Trace:
    data: dict[str, Any]

    def __init__(self, json_string: str):
        self.data = json.loads(json_string)

    def original_query(self) -> str:
        return self.data.get('original', 'MISSING DATA')

    def query_time(self) -> datetime:
        timestamp = self.data.get('time', 0)
        return datetime.fromtimestamp(timestamp)


server = Flask(__name__, template_folder='static')

TRACES_PATH = Path(__file__).parent.parent / 'traces'


def get_traces() -> Generator[str]:
    for filename in TRACES_PATH.iterdir():
        if filename.suffix == '.jsonl':
            with open(filename) as trace_file:
                yield from trace_file


TRACES = list(map(Trace, get_traces()))
TRACES.sort(key=Trace.query_time, reverse=True)


@server.get("/")
def main():
    return _list_traces(TRACES)


@server.get('/search/<query>')
def get_searched(query: str):
    return _list_traces(_search(query))


@server.get('/traces/<n>')
def get_trace(n: int):
    n = int(n)

    if 0 <= n < len(TRACES):
        svg = to_graphviz(TRACES[n].data).pipe(format='svg')
        return _render(svg)
    abort(404)


def _render(svg: bytes) -> str:
    return render_template('svg_viewport.html', svg=svg.decode())


def _list_traces(traces: list[Trace]):
    queries = [
        (
            trace.original_query(),
            trace.query_time().strftime('%Y-%m-%d %H:%M:%S'),
            n
        )
        for n, trace in enumerate(traces)
    ]
    return render_template('query_table.html', queries=queries)


def _search(query: str) -> list[Trace]:
    return [trace for trace in TRACES if query in trace.original_query()]


server.run('localhost', 5000)
