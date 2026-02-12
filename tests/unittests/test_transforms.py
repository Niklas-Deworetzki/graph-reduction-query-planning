import unittest

from grqe.query import *
from grqe.transformations import *

ATOM_A = Atom(0, 'A', 'a')
ATOM_B = Atom(0, 'B', 'b')
ATOM_C = Atom(0, 'C', 'c')
ATOM_D = Atom(0, 'D', 'd')
ATOM_E = Atom(0, 'E', 'e')
ATOM_F = Atom(0, 'F', 'f')
ATOM_G = Atom(0, 'G', 'g')

VARIABLE_WIDTH_NODE = Disjunction([
    Lookup([ATOM_E]),
    Lookup([ATOM_F, ATOM_G.shift(5)]),
])

FIXED_WIDTH_NODE = Disjunction(
    [Lookup([atom]) for atom in [ATOM_E, ATOM_F, ATOM_G]]
)


class ParserTest(unittest.TestCase):

    def test_leaf_fusion(self):
        test_cases = [
            (
                'Lookup nodes remain unchanged.',
                Lookup([ATOM_A, ATOM_B]),
                Lookup([ATOM_A, ATOM_B]),
            ),
            (
                'Lookups in conjunction are fused.',
                Conjunction([
                    Lookup([atom])
                    for atom in [ATOM_A, ATOM_B, ATOM_C]]
                ),
                Lookup([ATOM_A, ATOM_B, ATOM_C]),
            ),
            (
                'Nested conjunction is fused.',
                Conjunction([
                    Lookup([ATOM_A]),
                    Conjunction([
                        Lookup([atom])
                        for atom in [ATOM_B, ATOM_C]]
                    ),
                ]),
                Lookup([ATOM_A, ATOM_B, ATOM_C]),
            ),
            (
                'Unfusable elements in conjunction remain.',
                Conjunction([
                    Lookup([ATOM_A]),
                    Lookup([ATOM_B]),
                    FIXED_WIDTH_NODE,
                ]),
                Conjunction([
                    Lookup([ATOM_A, ATOM_B]),
                    FIXED_WIDTH_NODE,
                ]),
            ),
            (
                'Sequence is fused with proper offsets.',
                Sequence([
                    Lookup([ATOM_A]),
                    Lookup([ATOM_B]),
                    Lookup([ATOM_C]),
                ]),
                Lookup([ATOM_A, ATOM_B.shift(1), ATOM_C.shift(2)]),
            ),
            (
                'Unfusable sequence prefix is prepended to fused lookups.',
                Sequence([
                    FIXED_WIDTH_NODE,
                    FIXED_WIDTH_NODE,
                    Lookup([ATOM_A]),
                    Lookup([ATOM_B]),
                ]),
                Sequence([
                    FIXED_WIDTH_NODE,
                    FIXED_WIDTH_NODE,
                    Lookup([ATOM_A, ATOM_B.shift(1)]),
                ]),
            ),
            (
                'Unfusable sequence suffix is appended to fused lookups.',
                Sequence([
                    Lookup([ATOM_A]),
                    Lookup([ATOM_B]),
                    FIXED_WIDTH_NODE,
                    FIXED_WIDTH_NODE,
                ]),
                Sequence([
                    Lookup([ATOM_A, ATOM_B.shift(1)]),
                    FIXED_WIDTH_NODE,
                    FIXED_WIDTH_NODE,
                ]),
            ),
            (
                'Unfusable sequence prefix and suffix are around fused lookups.',
                Sequence([
                    FIXED_WIDTH_NODE,
                    FIXED_WIDTH_NODE,
                    FIXED_WIDTH_NODE,
                    Lookup([ATOM_A]),
                    Lookup([ATOM_B]),
                    FIXED_WIDTH_NODE,
                    FIXED_WIDTH_NODE,
                ]),
                Sequence([
                    FIXED_WIDTH_NODE,
                    FIXED_WIDTH_NODE,
                    FIXED_WIDTH_NODE,
                    Lookup([ATOM_A, ATOM_B.shift(1)]),
                    FIXED_WIDTH_NODE,
                    FIXED_WIDTH_NODE,
                ]),
            ),
            (
                'Unfusable sequence overlapping with leafs is turned into a conjunction.',
                Sequence([
                    Lookup([ATOM_A]),
                    FIXED_WIDTH_NODE,
                    Lookup([ATOM_B]),
                ]),
                Conjunction([
                    Sequence([
                        Arbitrary(),
                        FIXED_WIDTH_NODE,
                        Arbitrary(),
                    ]),
                    Lookup([ATOM_A, ATOM_B.shift(2)]),
                ])
            ),
            (
                'Unfusable sequence remains unchanged.',
                Sequence([
                    FIXED_WIDTH_NODE,
                    FIXED_WIDTH_NODE,
                    FIXED_WIDTH_NODE,
                ]),
                Sequence([
                    FIXED_WIDTH_NODE,
                    FIXED_WIDTH_NODE,
                    FIXED_WIDTH_NODE,
                ]),
            )
        ]

        for description, example_input, expected_output in test_cases:
            with self.subTest(msg=description):
                canonical_actual_output = canonical(fuse_leaves(example_input))
                canonical_expected_output = canonical(expected_output)

                self.assertEqual(canonical_expected_output, canonical_actual_output)
