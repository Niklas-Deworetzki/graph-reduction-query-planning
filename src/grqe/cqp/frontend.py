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

    attribute: KEY ATTRIBUTE_OPERATOR VALUE

    BINARY_OPERATOR: "|" | "&"
    SUFFIX_OPERATOR: "+" | "*" | "?"
    ATTRIBUTE_OPERATOR: "=" | "!="

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

            positive_atoms = []
            negative_atoms = []
            for is_positive, atom in map(to_atom, attributes):
                if is_positive:
                    positive_atoms.append(atom)
                else:
                    negative_atoms.append(atom)

            if positive_atoms:
                positive = Lookup(tuple(positive_atoms))
            else:
                positive = Arbitrary()

            if negative_atoms:
                negative = Lookup(tuple(negative_atoms))
                return Subtraction(positive, negative)
            else:
                return positive

        case _:
            singleton = query.children[0]
            return convert(singleton)


def to_atom(attribute: Parsed) -> tuple[bool, Atom]:
    key, op, value = attribute.children
    return op == '=', Atom(0, key, value[1:-1])


def to_span_atom(attribute: Parsed) -> tuple[bool, SpanAtom]:
    key, op, value = attribute.children
    return op == '=', SpanAtom(key, value[1:-1])


def to_span_query(span: str, metadata: Iterable[Parsed]) -> SpanLookup:
    positive_atoms = []
    negative_atoms = []
    for is_positive, atom in map(to_span_atom, metadata):
        if is_positive:
            positive_atoms.append(atom)
        else:
            negative_atoms.append(atom)

    positive = SpanLookup(span, positive_atoms)
    if negative_atoms:
        negative = SpanLookup(span, negative_atoms)
        return Subtraction(positive, negative)
    else:
        return positive


def parse(s: str) -> Node:
    try:
        tree = _parser.parse(s)
        unparsed_query = tree.children[0]

        query = convert(unparsed_query)
        if len(tree.children) > 1:
            span, *metadata = tree.children[1].children
            span_query = to_span_query(span, metadata)
            query = Contained(
                query,
                span_query,
            )

        return query
    except LarkError as err:
        raise ValueError(s) from err
