from typing import Type

from lark import Lark
from lark.exceptions import LarkError

from grqe.query import *

GRAMMAR = """
    start: statement+
    
    statement: IDENTIFIER "=" expression ";"
    
    expression: "(" expression ")"      -> parenthesized
              | OPERATOR expression+    -> application
              | IDENTIFIER              -> variable
              | "[" atom* "]"           -> lookup
    
    atom: KEY "@" OFFSET "=" STRING 

    KEY:        LETTER (LETTER | DIGIT)*
    OPERATOR:   LOWER (LOWER | DIGIT)*
    IDENTIFIER: UPPER (UPPER | DIGIT)*
    OFFSET:     "-"? DIGIT+
    
    %import common.WS
    %import common.DIGIT
    %import common.ESCAPED_STRING -> STRING
    %import common.LCASE_LETTER -> LOWER
    %import common.UCASE_LETTER -> UPPER
    %import common.LETTER
    
    %ignore WS
"""

_parser = Lark(GRAMMAR, parser='lalr')


class ParseException(Exception):
    pass


class UnknownOperatorException(ParseException):
    def __init__(self, operator: str):
        super().__init__(f'Operator `{operator}` is not supported.')


class UnknownVariableException(ParseException):
    def __init__(self, variable: str):
        super().__init__(f'Variable `{variable}` is not defined.')


class ReassignmentException(ParseException):
    def __init__(self, variable: str):
        super().__init__(f'Reassignment of variable `{variable}` is not allowed.')


class ArityException(ParseException):
    def __init__(self, operator: str, expected_arity: int, actual_arity: int):
        super().__init__(f'Operator `{operator}` has arity {expected_arity} but was provided {actual_arity} arguments.')


class Transform:
    constructors: ClassVar[dict[str, Type[Node]]] = {
        'con': Conjunction,
        'dis': Disjunction,
        'seq': Sequence,
        'alt': Alternative,
        'sub': Subtraction,
    }

    environment: dict[str, Node]

    def __init__(self):
        self.environment = {}

    def transform(self, t) -> Node:
        match t.data:
            case 'start':
                for statement in t.children:
                    res = self.transform(statement)
                return res

            case 'statement':
                id, expr = t.children
                value = self.transform(expr)

                if id in self.environment:
                    raise ReassignmentException(id)
                self.environment[id] = value
                return value

            case 'parenthesized':
                child, = t.children
                return self.transform(child)

            case 'application':
                operator, *args = t.children

                constructor = self.constructors.get(operator)
                if constructor is None:
                    raise UnknownOperatorException(operator)

                args = [self.transform(arg) for arg in args]
                if constructor.arity is not None:
                    if constructor.arity != len(args):
                        raise ArityException(operator, constructor.arity, len(args))
                    return constructor(*args)

                return constructor(tuple(args))

            case 'variable':
                value = self.environment.get(t.children[0])
                if value is None:
                    raise UnknownVariableException(t.children[0])
                return value

            case 'lookup':
                atoms = tuple(self.atom(t) for t in t.children)
                if len(atoms) == 0:
                    return Arbitrary()
                return Lookup(atoms)

        raise NotImplementedError()

    def atom(self, t) -> Atom:
        key, offset, value = t.children
        return Atom(int(offset), key, value[1:-1])


def parse(s: str) -> Node:
    try:
        tree = _parser.parse(s)
        return Transform().transform(tree)
    except LarkError as err:
        raise ValueError(s) from err
