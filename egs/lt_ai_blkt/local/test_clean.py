from unittest import TestCase

from egs.lt_ai_blkt.local.clean import skip
from egs.lt_ai_blkt.local.utils import Word


class Test(TestCase):
    def test_skip(self):
        self.assertTrue(skip(Word("123")))
        self.assertTrue(skip(Word("I(=M)")))
        self.assertTrue(skip(Word("II(=M)")))
        self.assertTrue(skip(Word("V(=M)")))
        self.assertTrue(skip(Word("mama;")))
        self.assertTrue(skip(Word("mama...")))

