from unittest import TestCase

from egs.lt_ai_blkt.local.data_module import split_text


class Test(TestCase):
    def test_split_text(self):
        words, case, puncts = split_text("Olia, kaip XX Jauties? AA.")
        self.assertEqual(words, ["olia", "kaip", "xx", "jauties", "aa"])
        self.assertEqual(case, [2, 0, 1, 2, 1])
        self.assertEqual(puncts, [1, 0, 0, 3, 2])
