import datetime
import json
from typing import Iterator, Optional
from xml.sax.saxutils import escape

import graphviz

_GRAPHVIZ_NODE_STYLE = {
    'shape': 'plaintext',
}

_GRAPHVIZ_EDGE_STYLE = {
    'dir': 'back',
}

_GRAPHVIZ_STYLE = {
}

MIN_COLOR = '#FBEF76'
MAX_COLOR = '#FA5C5C'
NO_DATA_COLOR = '#CCCCCC'

ROOT_NODE_COLOR = '#FFFFFF'

HIGHLIGHTING_FIELD = 'time'

SPECIAL_FIELDS = ['time', 'size']


def to_graphviz(data: str | dict) -> graphviz.Digraph:
    if isinstance(data, str):
        data = json.loads(data)
    data: dict

    profiling_info: dict[str, dict] = data['profiling_info']
    highlighting_values = [
        int(info[HIGHLIGHTING_FIELD])
        for info in profiling_info.values()
        if HIGHLIGHTING_FIELD in info
    ]

    max_value = max(highlighting_values) if highlighting_values else None
    min_value = min(highlighting_values) if highlighting_values else None

    when = datetime.datetime.fromtimestamp(data['time']).strftime('%Y-%m-%d %H:%M:%S')
    graph = graphviz.Digraph(
        comment=f'Executed at {when}',
        node_attr=_GRAPHVIZ_NODE_STYLE | _GRAPHVIZ_STYLE,
        edge_attr=_GRAPHVIZ_EDGE_STYLE | _GRAPHVIZ_STYLE,
    )

    root_label = _format_values_to_html_label(data['original'], data['meta_profiling_info'], ROOT_NODE_COLOR)
    graph.node('0', root_label)

    for node_key, representation in data['nodes'].items():
        profiling: dict = profiling_info[node_key]

        node_color = _interpolate_color(min_value, max_value, profiling.get(HIGHLIGHTING_FIELD))
        label = _format_values_to_html_label(representation, profiling, node_color)
        graph.node(node_key, label)

    for source, targets in data['edges'].items():
        for target in targets:
            graph.edge(target, source)
    graph.edge('0', data['root'])
    return graph


def _format_values_to_html_label(representation: str, info: dict, color: str) -> str:
    rows = [
        f'<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="4" BGCOLOR="{color}">',
        f'<TR><TD COLSPAN="2"><B>{escape(representation)}</B></TD></TR>',
    ]
    for key, value in _iterate_entries(info):
        rows.append(
            f'<TR><TD ALIGN="left">{key}</TD><TD ALIGN="right">{value}</TD></TR>'
        )
    rows.append('</TABLE>')

    inner_html = ''.join(rows)
    return f'<{inner_html}>'


def _iterate_entries(info: dict) -> Iterator[tuple[str, str]]:
    fields = SPECIAL_FIELDS + sorted(set(info.keys()) - set(SPECIAL_FIELDS))
    for field in fields:
        value = info.get(field)
        if isinstance(value, int):
            value = format_time(value)
        if value is not None:
            yield escape(field), escape(value)


def format_time(ns: int) -> str:
    if ns < 1000:
        return f'{ns}ns'
    if ns < 1000 * 1000:
        return '%.2fus' % (ns / 1000)
    if ns < 1000 * 1000 * 1000:
        return '%.2fms' % (ns / 1000 / 1000)
    return '%.2fs' % (ns / 1000 / 1000 / 1000)


def format_bytesize(n: int) -> str:
    for prefix in ['', 'k', 'M', 'G']:
        if n < 1024:
            break
        n //= 1024
    return f'{n} {prefix}b'


def _interpolate_color(val_min: int, val_max: int, value: Optional[int]) -> str:
    if value is None:
        return NO_DATA_COLOR

    if val_min == val_max:
        return MAX_COLOR

    res = '#'
    for channel in range(3):
        chan_min = int(MIN_COLOR[1 + 2 * channel: 1 + 2 * (channel + 1)], 16)
        chan_max = int(MAX_COLOR[1 + 2 * channel: 1 + 2 * (channel + 1)], 16)

        perc = (value - val_min) / (val_max - val_min)
        chan_val = perc * chan_max + (1 - perc) * chan_min
        chan_val = round(chan_val)
        res += f'{chan_val:0>2X}'
    return res
