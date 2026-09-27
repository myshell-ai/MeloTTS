import unittest

from melo.split_utils import split_sentence, txtsplit


class SplitUtilsTest(unittest.TestCase):
    def test_decimal_and_thousands_stay_one_number(self):
        self.assertEqual(
            split_sentence("The value is 3.14 today.", language_str="EN"),
            ["The value is 3.14 today."],
        )
        self.assertEqual(
            split_sentence("It costs 1,000 dollars today.", language_str="EN"),
            ["It costs 1,000 dollars today."],
        )
        self.assertEqual(
            split_sentence("版本1.2.3发布了。", language_str="ZH"),
            ["版本1.2.3发布了."],
        )

    def test_sentence_punctuation_still_splits(self):
        self.assertEqual(txtsplit("Hello.World", 20, 40), ["Hello. World"])
        parts = txtsplit("One two. Three four five six seven.", 12, 24)
        self.assertEqual(parts[0], "One two.")
        self.assertIn("seven.", parts[-1])


if __name__ == "__main__":
    unittest.main()
