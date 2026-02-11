import html
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, ClassVar, Iterable

XML_NAME_START_CHAR = r'[_:a-zA-Z\u2070-\u218F\u2C00-\u2FEF\u3001-\uD7FF\uF900-\uFDCF\uFDF0-\uFFFD]'
XML_NAME_CHAR = fr'(?:{XML_NAME_START_CHAR}|[\-.0-9\u00B7\u0300-\u036F\u203F-\u2040])'
XML_NAME = fr'(?:{XML_NAME_START_CHAR}{XML_NAME_CHAR}*)'

XML_ATTRIBUTE_REGEX = re.compile(fr'({XML_NAME}) *= *"([^"]*)"')


def extract_opening_tag(tag: str) -> tuple[str, dict[str, bytes]]:
    end_of_name = tag.find(' ')
    if end_of_name == -1:
        end_of_tag = tag.rfind('>')
        return tag[1:end_of_tag], {}
    else:
        attributes = {
            match.group(1): html.unescape(match.group(2).encode())
            for match in XML_ATTRIBUTE_REGEX.finditer(tag, end_of_name)
        }
        return tag[1:end_of_name], attributes


@dataclass(frozen=True)
class CorpusHandler:
    on_token: Callable[[int, dict[str, bytes]], None]
    on_span: Callable[[str, int, int, dict[str, bytes]], None]


@dataclass(frozen=True)
class VrtParser:
    NO_VALUE: ClassVar[bytes] = b''

    columns: list[str]
    spans: dict[str, set[str]]

    def parse(
            self,
            callback: CorpusHandler,
            paths: Iterable[Path],
            included_filetypes: set[str] = frozenset({'.vrt'})
    ) -> int:
        def collect_corpus_files():
            for path in paths:
                if path.is_file():
                    yield path

                else:
                    for (dirpath, _, filenames) in os.walk(path):
                        for filename in filenames:
                            if any(filename.endswith(suffix) for suffix in included_filetypes):
                                yield Path(os.path.join(dirpath, filename))

        token_count = 0
        for file in collect_corpus_files():
            token_count += self._parse_vrt(file, token_count, callback)
        return token_count

    def _parse_vrt(self, file: Path, token_count: int, callback: CorpusHandler) -> int:
        open_spans: list[str] = []
        open_span_data: list[tuple[int, dict[str, bytes]]] = []

        with file.open('r') as f:
            for lineno, content in enumerate(f, start=1):
                if content.isspace():
                    continue

                if content.startswith('<'):
                    closing_bracket = content.rfind('>')
                    match content[1]:
                        case '!':  # <!-- xml comment -->
                            continue
                        case '?':  # <?xml processing instruction>
                            continue

                        case '/':  # </SPAN>
                            span_name = content[2:closing_bracket].strip()
                            if open_spans[-1] != span_name:
                                continue  # End of a region not indexed.

                            open_spans.pop()
                            start, attributes = open_span_data.pop()

                            extracted_attributes = {
                                key: attributes.get(key, VrtParser.NO_VALUE)
                                for key in self.spans[span_name]
                            }
                            callback.on_span(span_name, start, token_count, extracted_attributes)

                        case _:
                            span_name, attributes = extract_opening_tag(content)

                            if span_name in self.spans:
                                open_spans.append(span_name)
                                open_span_data.append((token_count, attributes))

                else:
                    column_values = html.unescape(content.rstrip('\n\r')).split('\t')
                    attributes = {
                        column: value.encode()
                        for column, value in zip(self.columns, column_values)
                    }
                    callback.on_token(token_count, attributes)
                    token_count += 1
        return token_count
