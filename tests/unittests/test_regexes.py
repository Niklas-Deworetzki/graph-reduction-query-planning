import unittest

from grqe import regexes

class RegexesTest(unittest.TestCase):

    def test_literal_positive(self):
        self.assertTrue(regexes.is_literal('abc'))

    def test_literal_negative(self):
        self.assertFalse(regexes.is_literal('abc?'))
        self.assertFalse(regexes.is_literal('abc+'))
        self.assertFalse(regexes.is_literal('abc*'))
        self.assertFalse(regexes.is_literal('ab|c'))
        self.assertFalse(regexes.is_literal('a[bc]'))

    def test_symboltable_unchanged(self):
        pattern = regexes.for_symboltable(b'abc')
        matches = pattern.findall(b'\x00a\x00ab\x00abc\x00')
        self.assertEqual(1, len(matches))


    def test_symboltable_respects_boundaries(self):
        pattern = regexes.for_symboltable(b'a.*')
        matches = pattern.findall(b'\x00a\x00ab\x00abc\x00')
        self.assertEqual(3, len(matches))

    def test_symboltable_does_not_change_dot(self):
        pattern = regexes.for_symboltable(b'a\\.')
        matches = pattern.findall(b'\x00a\x00a.\x00ab\x00')
        self.assertEqual(1, len(matches))

