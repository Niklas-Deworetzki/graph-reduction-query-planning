from lark import Lark, Token, Tree
from lark.exceptions import LarkError

from grqe.query import *

GRAMMAR = """
    start: bin within?
    
    within: "within" KEY ("where" attribute ("," attribute)*)?
    
    bin:    seq BINARY_OPERATOR seq             -> binary
       |    seq
    seq:    pat+                                -> sequence
    pat:    tok SUFFIX_OPERATOR?                -> pattern
    tok: "[" attributes "]"                     -> token
       | "(" bin ")"

    attributes: (attribute ("," attribute)*) ?

    attribute: KEY "=" VALUE

    BINARY_OPERATOR: "|" | "&"
    SUFFIX_OPERATOR: "+" | "*" | "?"    

    KEY:        LETTER (LETTER | DIGIT)*
    %import common.ESCAPED_STRING -> VALUE

    %import common.WS
    %import common.DIGIT
    %import common.LETTER

    %ignore WS
"""

_parser = Lark(GRAMMAR, parser='lalr')

type Parsed = Tree[Token]

_BINARY_CONSTRUCTORS = {
    '&': Conjunction,
    '|': Disjunction,
}


def convert(query: Parsed) -> Node:
    match query.data:
        case 'binary':
            lhs, op, rhs = query.children

            operands = [convert(lhs), convert(rhs)]
            return _BINARY_CONSTRUCTORS[op](operands)

        case 'sequence':
            translated: list[Node] = []
            for element in query.children:

                child = convert(element.children[0])
                suffix = element.children[1] if len(element.children) > 1 else None

                if suffix in {'*', '+'}:
                    child = Repeat(child)
                if suffix in {'*', '?'}:
                    child = Disjunction([child, Epsilon()])

                translated.append(child)
            return Sequence(translated)

        case 'token':
            attributes = query.children[0].children
            if len(attributes) == 0:
                return Arbitrary()

            atoms = map(to_atom, attributes)
            return Lookup(tuple(atoms))

        case _:
            singleton = query.children[0]
            return convert(singleton)


def to_atom(attribute: Parsed) -> Atom:
    key, value = attribute.children
    return Atom(0, key, value[1:-1])


def to_span_atom(attribute: Parsed) -> SpanAtom:
    key, value = attribute.children
    return SpanAtom(key, value[1:-1])


def parse(s: str) -> Node:
    try:
        tree = _parser.parse(s)
        unparsed_query = tree.children[0]

        query = convert(unparsed_query)
        if len(tree.children) > 1:
            span, *metadata = tree.children[1].children

            span_query = SpanLookup(
                span,
                map(to_span_atom, metadata)
            )
            query = Contained(
                query,
                span_query,
            )

        return query
    except LarkError as err:
        raise ValueError(s) from err
