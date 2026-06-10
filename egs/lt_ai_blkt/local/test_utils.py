from unittest import TestCase

from egs.lt_ai_blkt.local.utils import split_word_punctuation


class Test(TestCase):
    def test_split_word_punctuation(self):
        w, p = split_word_punctuation("olia,")
        self.assertEqual(w, "olia")
        self.assertEqual(p, ",")

        w, p = split_word_punctuation("olia...")
        self.assertEqual(w, "olia")
        self.assertEqual(p, "...")

        w, p = split_word_punctuation("olia...a")
        self.assertEqual(w, "olia")
        self.assertEqual(p, "...a")

        w, p = split_word_punctuation(",")
        self.assertEqual(w, "")
        self.assertEqual(p, ",")