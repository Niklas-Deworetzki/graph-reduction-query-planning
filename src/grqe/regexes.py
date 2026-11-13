import re

import greenery
from greenery.rxelems import Conc, Mult, Pattern


def is_literal(regex: str) -> bool:
    try:
        return greenery.parse(regex).cardinality() == 1
    except OverflowError:
        return False


DOT_CHARCLASS = ~greenery.Charclass(())
NOT_NUL_CHARCLASS = ~greenery.Charclass('\x00')


def for_symboltable(regex: bytes) -> re.Pattern[bytes]:
    pattern = greenery.parse(regex.decode())

    def recur(p):
        if p == DOT_CHARCLASS:
            return NOT_NUL_CHARCLASS

        elif isinstance(p, Pattern):
            return Pattern(*map(recur, p.concs))

        elif isinstance(p, Conc):
            return Conc(*map(recur, p.mults))

        elif isinstance(p, Mult):
            return Mult(recur(p.multiplicand), p.multiplier)

        return p

    adjusted = str(recur(pattern)).encode()
    adjusted = b'\x00' + adjusted + b'(?=\x00)'
    return re.compile(adjusted)
