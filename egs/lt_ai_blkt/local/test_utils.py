from unittest import TestCase

from egs.lt_ai_blkt.local.utils import split_word_punctuation, Word


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


class TestWord(TestCase):
    def test_init_word(self):
        w = Word("olia(=M)")
        self.assertEqual(w.word, "olia")
        self.assertEqual(w.mi, "M")
        self.assertEqual(w.punct, "")


        w = Word("olia(=M).")
        self.assertEqual(w.word, "olia")
        self.assertEqual(w.punct, ".")
        self.assertEqual(w.mi, "M")


        w = Word("olia")
        self.assertEqual(w.word, "olia")
        self.assertEqual(w.mi, "")
        self.assertEqual(w.punct, "")

        w = Word("olia.(=M)")
        self.assertEqual(w.word, "olia")
        self.assertEqual(w.mi, "M")
        self.assertEqual(w.punct, ".")

        w = Word("olia.,-")
        self.assertEqual(w.word, "olia")
        self.assertEqual(w.punct, ".,-")

        w = Word("-")
        self.assertEqual(w.word, "")
        self.assertEqual(w.punct, "-")

    def test_to_str(self):
            w = Word("olia(=M)")
            self.assertEqual(w.to_str(), "olia(=M)")

            w = Word("olia(=M).")
            self.assertEqual(w.to_str(), "olia.(=M)")

            w = Word("olia")
            self.assertEqual(w.to_str(), "olia")

            w = Word("olia.(=M)")
            self.assertEqual(w.to_str(), "olia.(=M)")

            w = Word("olia.,-")
            self.assertEqual(w.to_str(), "olia.,-")

            w = Word("-")
            self.assertEqual(w.to_str(), "-")
