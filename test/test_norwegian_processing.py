import unittest
import os
import sys

# Add project root to path to allow importing melo
# This might be needed if tests are run from the 'test' directory directly
# Or if the project isn't installed in the environment
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from melo.text.norwegian import text_normalize, g2p
from melo.text import symbols # To access UNK, pad etc.

# Attempt to import espeak_ng to check availability
espeak_ng_available = False
try:
    import subprocess
    subprocess.run(['espeak-ng', '--version'], capture_output=True, check=True)
    espeak_ng_available = True
except (FileNotFoundError, subprocess.CalledProcessError):
    print("Warning: espeak-ng not found or not working. G2P tests will be skipped.")


class TestNorwegianTextProcessing(unittest.TestCase):

    def test_text_normalize_lowercase_and_cleaning(self):
        text = "Dette  Er — En Test.“Med Æ, Ø, Å.”"
        expected = "dette er - en test.\"med æ, ø, å.\"" # lowercased, whitespace norm, em-dash, quote norm
        self.assertEqual(text_normalize(text), expected)

    def test_text_normalize_abbreviations(self):
        test_cases = {
            "f.eks. dette": "for eksempel dette",
            "Det er osv.": "det er og så videre",
            "Bl.a. epler": "blant annet epler",
            "Ca. 100": "sirka 100", # Number will be normalized in next step by text_normalize
            "dvs. at det...": "det vil si at det...",
            "3. etg.": "3. etasje", # number part for etg. needs ordinal, current rule is basic
            "nr. 1 er best": "nummer 1 er best",
            "Ring tlf. 123": "ring telefon 123",
            "se fig. 2": "se figur 2",
            "les kap. 3": "les kapittel 3",
            "Ta med pga. regn": "ta med på grunn av regn",
            "kl. 14:00": "klokka 14:00" # time parsing not yet implemented after kl.
        }
        for text, expected_expansion in test_cases.items():
            # Test abbreviation part, number part will be handled by number normalizer later in text_normalize
            # So, we check the expanded form before number normalization for some cases.
            # For "Ca. 100", after abbr exp: "sirka 100". After num norm: "sirka hundre"
            # For "3. etg.", after abbr exp: "3. etasje". Num norm would make it "tre. etasje" (incorrect)
            # This highlights interaction between rules.
            # For now, testing the direct output of _expand_abbreviations_no via text_normalize (lowercase effect)

            # Let's test the final output of text_normalize for some simple cases
            if "100" in expected_expansion: # "Ca. 100" -> "sirka hundre"
                 normalized_output = text_normalize(text)
                 self.assertEqual(normalized_output, expected_expansion.replace("100", "hundre").lower())
            elif "3." in expected_expansion: # "3. etg." -> "3. etasje" -> "tre. etasje" (current basic)
                 normalized_output = text_normalize(text)
                 # current basic year/int rule might make "3." into "tre."
                 self.assertEqual(normalized_output, expected_expansion.replace("3.", "tre.").lower())
            elif "123" in expected_expansion: # "Ring tlf. 123" -> "ring telefon hundre og tjue tre"
                 normalized_output = text_normalize(text)
                 self.assertEqual(normalized_output, expected_expansion.replace("123", "hundre og tjue tre").lower())
            elif "2" in expected_expansion and "fig" in text: # "se fig. 2" -> "se figur to"
                 normalized_output = text_normalize(text)
                 self.assertEqual(normalized_output, expected_expansion.replace("2", "to").lower())
            elif "3" in expected_expansion and "kap" in text: # "les kap. 3" -> "les kapittel tre"
                 normalized_output = text_normalize(text)
                 self.assertEqual(normalized_output, expected_expansion.replace("3", "tre").lower())
            elif "14:00" in expected_expansion: # "kl. 14:00" -> "klokka 14:00" (time not parsed yet)
                 normalized_output = text_normalize(text)
                 self.assertEqual(normalized_output, expected_expansion.lower()) # Time part remains
            else:
                 self.assertEqual(text_normalize(text), expected_expansion.lower())


    def test_text_normalize_numbers(self):
        test_cases = {
            "året er 1995": "året er nitten nittifem",
            "vi er i 2023": "vi er i to tusen og tjue tre",
            "tallet 123": "tallet hundre og tjue tre",
            "1,23 prosent": "en komma tjue tre prosent", # Assumes num after comma is read as whole
            "prisen er 100 kr.": "prisen er hundre kroner",
            "det koster kr 50": "det koster femti kroner",
            "5 kg": "fem kg", # kg not expanded, only number
            "2000": "to tusen", # Year rule
            "1900": "nitten hundre" # Year rule
        }
        for text, expected_words in test_cases.items():
            self.assertEqual(text_normalize(text), expected_words)

    def test_text_normalize_combined(self):
        text = "Det var ca. 100 stk. f.eks. i år 1990."
        # Lowercase: "det var ca. 100 stk. f.eks. i år 1990."
        # Abbr: "det var sirka 100 stk. for eksempel i år 1990." (stk. not defined yet)
        # Nums: "det var sirka hundre stk. for eksempel i år nitten nitti."
        expected = "det var sirka hundre stk. for eksempel i år nitten nitti."
        self.assertEqual(text_normalize(text), expected)

        text_2 = "Nr. 1 koster kr. 50,5. Dette var i 2007."
        # Lower: "nr. 1 koster kr. 50,5. dette var i 2007."
        # Abbr: "nummer 1 koster kroner 50,5. dette var i 2007." (kr. -> kroner if not by num norm)
        # Num: "nummer en koster femti komma fem kroner. dette var i to tusen og sju." (order of kr might matter)
        # Current _normalize_numbers_no for kr: "(\d+[\.,]?\d*)\s*kr\.?" -> num + " kroner"
        # So "nummer en koster 50,5 kroner. ..." -> "nummer en koster femti komma fem kroner. ..."
        expected_2 = "nummer en koster femti komma fem kroner. dette var i to tusen og sju."
        self.assertEqual(text_normalize(text_2), expected_2)


    @unittest.skipUnless(espeak_ng_available, "espeak-ng not available, skipping g2p tests")
    def test_g2p_simple_word(self):
        text = "Hallo" # Example word
        # Assuming espeak-ng -v no --ipa=3 -q "Hallo" might give something like "hˈaloː" or "hˈɑloː" (actual output needed)
        # For this test, we are checking structure and that phonemes are valid symbols or UNK.
        phones, tones, word2ph = g2p(text, pad_start_end=False)

        self.assertTrue(len(phones) > 0, f"Phones list should not be empty for '{text}', got {phones}")
        self.assertEqual(len(phones), len(tones), "Phones and tones lists should have the same length")
        self.assertTrue(sum(word2ph) >= len(phones), "Sum of word2ph should be consistent with phone length")
        self.assertTrue(all(isinstance(p, str) for p in phones), "All phonemes should be strings")
        self.assertTrue(all(t == 0 for t in tones), "All tones should currently be 0") # Current implementation
        self.assertTrue(all(isinstance(w2p, int) for w2p in word2ph), "All word2ph entries should be integers")

        # Check if all output phonemes are known symbols (or UNK)
        # This relies on `no_symbols` in `symbols.py` being accurate for espeak's output
        for p in phones:
            self.assertTrue(p == symbols.UNK or p in symbols.symbols, f"Phoneme '{p}' not in symbols list and not UNK.")

        # Example of a more specific check if actual espeak output is known:
        # expected_phonemes_example = ['h', 'ˈ', 'a', 'l', 'o', 'ː'] # Hypothetical
        # self.assertEqual(phones, expected_phonemes_example, f"Phonemes for '{text}' did not match expected. Got: {phones}")
        # print(f"Phones for '{text}': {phones}")


    @unittest.skipUnless(espeak_ng_available, "espeak-ng not available, skipping g2p tests")
    def test_g2p_with_norwegian_chars(self):
        text = "blåbærsyltetøy" # Contains æ, ø, å
        phones, tones, word2ph = g2p(text, pad_start_end=True)

        self.assertTrue(len(phones) > 2, "Phones list should include pads and content")
        self.assertEqual(phones[0], symbols.pad, "First phoneme should be pad")
        self.assertEqual(phones[-1], symbols.pad, "Last phoneme should be pad")
        self.assertEqual(len(phones), len(tones), "Phones and tones lists should have the same length")
        for p in phones[1:-1]: # Check content phonemes
             self.assertTrue(p == symbols.UNK or p in symbols.symbols, f"Phoneme '{p}' not in symbols list and not UNK for '{text}'. Phones: {phones}")
        # print(f"Phones for '{text}': {phones}")

    @unittest.skipUnless(espeak_ng_available, "espeak-ng not available, skipping g2p tests")
    def test_g2p_sentence(self):
        text = "Dette er en test."
        phones, tones, word2ph = g2p(text)

        self.assertTrue(len(phones) > 2) # Should have content and pads
        self.assertEqual(len(phones), len(tones))
        for p in phones[1:-1]:
             self.assertTrue(p == symbols.UNK or p in symbols.symbols, f"Phoneme '{p}' not in symbols list and not UNK for '{text}'. Phones: {phones}")
        # print(f"Phones for '{text}': {phones}")

    @unittest.skipUnless(espeak_ng_available, "espeak-ng not available, skipping g2p tests")
    def test_g2p_unknown_word_produces_unk_or_fallback(self):
        text = "Xyzzyq" # Highly likely unknown
        phones, _, _ = g2p(text, pad_start_end=False)

        # Expected: _phonemize_espeak_no returns [symbols.UNK] for this.
        # Or if it somehow produces phonemes, they should map to UNK if not in no_symbols.
        # Or the g2p fallback `[symbols.UNK] * len(word)` might trigger.
        # Given `_phonemize_espeak_no` returns `[symbols.UNK]` on error/no output:
        if phones == [symbols.UNK]: # This is the most likely output from _phonemize_espeak_no
            pass
        elif all(p == symbols.UNK for p in phones) and len(phones) == len(text): # Fallback in g2p
            pass
        else:
            self.fail(f"Expected UNK phonemes for unknown word '{text}', got {phones}")

    def test_g2p_padding(self):
        text = "test"
        # Test default padding
        phones_padded, tones_padded, word2ph_padded = g2p(text, pad_start_end=True)
        self.assertEqual(phones_padded[0], symbols.pad)
        self.assertEqual(phones_padded[-1], symbols.pad)
        self.assertEqual(tones_padded[0], 0) # Assuming pad tone is 0
        self.assertEqual(tones_padded[-1], 0)
        self.assertEqual(word2ph_padded[0], 1) # Assuming pad w2p is 1
        self.assertEqual(word2ph_padded[-1], 1)

        # Test no padding
        phones_unpadded, tones_unpadded, word2ph_unpadded = g2p(text, pad_start_end=False)
        if phones_unpadded: # Only check if phonemes were produced
            self.assertNotEqual(phones_unpadded[0], symbols.pad)
            self.assertNotEqual(phones_unpadded[-1], symbols.pad)


if __name__ == '__main__':
    unittest.main()

# Note:
# These tests are basic and will need to be expanded.
# Crucially, for g2p tests to be more meaningful, `no_symbols` in `melo/text/symbols.py`
# needs to be updated with the actual IPA phonemes produced by `espeak-ng -v no --ipa=3`.
# Otherwise, many phonemes might be incorrectly mapped to UNK or tests might not reflect true g2p quality.
#
# The tests also rely on `espeak-ng` being installed and the 'no' voice data being available.
# If `espeak-ng` is not found, g2p tests are skipped.
#
# The BERT feature extraction (`get_bert_feature`) is not tested here as it's currently a placeholder
# in `norwegian.py` and depends on `norwegian_bert.py` which is not yet created.
# Tests for BERT features should be added once that part is implemented.
#
# To run this test file:
# python -m unittest test/test_norwegian_processing.py
# (Or your IDE's test runner, ensuring the project root is in PYTHONPATH)
