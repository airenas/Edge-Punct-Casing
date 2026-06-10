from unittest import TestCase

from egs.lt_ai_blkt.local.case import LOWER, get_case_id, CAP, UPPER, MIX_CASE


class Test(TestCase):
    def test_get_case_id(self):
        self.assertEqual(get_case_id("hello"), LOWER)
        self.assertEqual(get_case_id("Hello"), CAP)
        self.assertEqual(get_case_id("HELLO"), UPPER)
        self.assertEqual(get_case_id("HeLlo"), MIX_CASE)

        self.assertEqual(get_case_id("ŽĄSIS"), UPPER)
        self.assertEqual(get_case_id("Ž"), CAP)

