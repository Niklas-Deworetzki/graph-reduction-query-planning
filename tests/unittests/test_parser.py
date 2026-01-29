import unittest

from grqe.parser import *


class ParserTest(unittest.TestCase):

    def test_lookups(self):
        parse(
            """
            A = [x@0 = "abc"   y@1 = "äasd"   z@2 = "1231023123" ] ;
            B = [X@-1 = "" y@1111111111 = "" z0@0 = "" ] ;
            """
        )

    def test_operators(self):
        for operator in ('con', 'dis', 'seq', 'alt'):
            for i in range(1, 10):
                args = ' []' * i
                parse(f'A = {operator}{args};')

        parse('A = sub [] [];')
        parse('A = neg [];')

    def test_operator_arities(self):
        operators = {
            'sub': 2,
            'neg': 1,
        }
        for operator, arity in operators.items():
            for i in range(1, 10):
                if arity != i:
                    with self.assertRaises(ArityException):
                        args = ' []' * i
                        parse(f'A = {operator}{args};')

    def test_unknown_operator(self):
        with self.assertRaises(UnknownOperatorException):
            parse('A = unknown [] [];')

    def test_unknown_identifier(self):
        with self.assertRaises(UnknownVariableException):
            parse('A = A;')

    def test_assignment_order(self):
        parse('A = [] ; B = A;')

    def test_re_assignment(self):
        with self.assertRaises(ReassignmentException):
            parse('A = []; A = [];')

    def test_dag(self):
        res = parse('A = [] ; B = con A A A A A;')

        self.assertIsInstance(res, Conjunction)
        for element in res.elements:
            self.assertIs(element, res.elements[0])
