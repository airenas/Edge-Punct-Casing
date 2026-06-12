from unittest import TestCase

from egs.lt_ai_blkt.local.clean import skip, drop_symbols, fix_symbols
from egs.lt_ai_blkt.local.utils import Word


def from_str(param):
    strs = param.split()
    return [Word(s) for s in strs]


def to_str(param):
    return " ".join([w.to_str() for w in param])


class Test(TestCase):
    def test_skip(self):
        self.assertTrue(skip(Word("123")))
        self.assertTrue(skip(Word("I(=M)")))
        self.assertTrue(skip(Word("II(=M)")))
        self.assertTrue(skip(Word("V(=M)")))
        self.assertTrue(skip(Word("mama;")))
        self.assertTrue(skip(Word("mama...")))

    def test_drop_symbols(self):
        self.assertEqual(drop_symbols(""), "")
        self.assertEqual(drop_symbols("olia"), "olia")
        self.assertEqual(drop_symbols("olia,."), "olia,.")
        self.assertEqual(drop_symbols("`\"olia\"`"), "olia")
        self.assertEqual(drop_symbols("`\"olia\"`, `\"olia\"`, "), "olia , olia ,")

    def test_fix_symbols(self):
        self.assertEqual("olia, olia.", to_str(fix_symbols(from_str("olia ,.; olia.,,"))))
        self.assertEqual(to_str(fix_symbols(from_str("olia olia"))), "olia olia")
        self.assertEqual(to_str(fix_symbols(from_str("olia, olia."))), "olia, olia.")
        self.assertEqual("olia, holia.", to_str(fix_symbols(from_str("olia,.; holia.,,"))))
        self.assertEqual("olia— holia.", to_str(fix_symbols(from_str("olia - holia.,,"))))
        self.assertEqual("— olia— holia.", to_str(fix_symbols(from_str("— olia - — holia.,,"))))
