import os
import re
import subprocess
from transformers import AutoTokenizer

from . import symbols # To access pad, UNK etc.

import num2words # For number to word conversion

# --- Text Normalization ---

# Common Norwegian abbreviations (Bokmål)
# Using (pattern, replacement) tuples for more control with regex if needed later
_abbreviations_no_list = [
    (re.compile(r'\bf\.eks\.', re.IGNORECASE), "for eksempel"),
    (re.compile(r'\bosv\.', re.IGNORECASE), "og så videre"),
    (re.compile(r'\bbl\.a\.', re.IGNORECASE), "blant annet"),
    (re.compile(r'\bca\.', re.IGNORECASE), "sirka"),
    (re.compile(r'\bdvs\.', re.IGNORECASE), "det vil si"),
    (re.compile(r'\betg\.', re.IGNORECASE), "etasje"), # e.g., 3. etg. -> tredje etasje (ordinal needed)
    (re.compile(r'\bevt\.', re.IGNORECASE), "eventuelt"),
    (re.compile(r'\binkl\.', re.IGNORECASE), "inkludert"),
    # (re.compile(r'\bkr\.', re.IGNORECASE), "kroner"), # Often handled by number normalization
    (re.compile(r'\bmoh\.', re.IGNORECASE), "meter over havet"),
    (re.compile(r'\bnr\.', re.IGNORECASE), "nummer"),
    (re.compile(r'\bobs\.', re.IGNORECASE), "observer"), # or "NB" -> "Enn Bæ"
    (re.compile(r'\bpga\.', re.IGNORECASE), "på grunn av"),
    (re.compile(r'\bs\.', re.IGNORECASE), "side"), # e.g., s. 23 -> side tjue tre
    (re.compile(r'\btlf\.', re.IGNORECASE), "telefon"),
    (re.compile(r'\badr\.', re.IGNORECASE), "adresse"),
    (re.compile(r'\bfig\.', re.IGNORECASE), "figur"),
    (re.compile(r'\bkap\.', re.IGNORECASE), "kapittel"),
    (re.compile(r'\bmvh\.', re.IGNORECASE), "med vennlig hilsen"), # Less likely in TTS input text
    (re.compile(r'\bkl\.', re.IGNORECASE), "klokka"), # e.g. kl. 14 -> klokka fjorten
]

def _expand_abbreviations_no(text):
    for pattern, replacement in _abbreviations_no_list:
        text = pattern.sub(replacement, text)
    return text

# Helper for number normalization
def _norwegian_number_to_words(n_str, to_type='cardinal'):
    try:
        # num2words for Norwegian might have issues with very large numbers or specific formatting.
        # For ordinals, num2words might output "første.", "andre.", so strip trailing period.
        if not n_str.isdigit(): # Handle cases like "1." for ordinals if passed directly
            n_str = re.sub(r'\D', '', n_str)
            if not n_str: return "" # return original if not a number after stripping

        n = int(n_str)
        if to_type == 'ordinal':
            # num2words ordinal output for 'no' can be like "1ste", "2dre", or "første.", "andre."
            # We want full words.
            # For 'no', num2words might produce "1.", "2." for some ordinals, or "første", "andre"
            # It seems num2words 'no' ordinal output is inconsistent for simple use.
            # Let's test its behavior or use a custom map for common ordinals if needed.
            # Example: num2words.num2words(1, lang='no', to='ordinal') -> 'første'
            # num2words.num2words(3, lang='no', to='ordinal') -> 'tredje'
            # num2words.num2words(10, lang='no', to='ordinal') -> 'tiende'
            word = num2words.num2words(n, lang='no', to='ordinal')
        elif to_type == 'year':
            # e.g. 1995 -> "nitten nittifem"
            #      2005 -> "to tusen og fem"
            #      2023 -> "to tusen og tjue tre"
            if 1000 <= n <= 2099: # Common year range
                if n % 100 == 0 and n % 1000 != 0 : # 1900, 2000
                     word = num2words.num2words(n, lang='no') # "nitten hundre", "to tusen"
                elif n > 1999: # 2000-2099
                    word = num2words.num2words(n, lang='no') # "to tusen og fem"
                else: # 1xxx years, read as two pairs typically up to 1999
                    first_pair = n // 100
                    second_pair = n % 100
                    word_fp = num2words.num2words(first_pair, lang='no')
                    if second_pair == 0:
                        word_sp = "hundre" # e.g. 1900 -> nitten hundre
                        return f"{word_fp} {word_sp}"
                    elif second_pair < 10 and first_pair > 0 : # e.g. 1905 -> nitten null fem (or nitten hundre og fem)
                        # num2words handles "nitten hundre og fem" better.
                         word = num2words.num2words(n, lang='no')
                    else:
                        word_sp = num2words.num2words(second_pair, lang='no')
                        word = f"{word_fp} {word_sp}" # "nitten nittifem"
            else: # Other years, or general numbers
                word = num2words.num2words(n, lang='no')
        else: # cardinal
            word = num2words.num2words(n, lang='no')

        return word.replace('-', ' ') # Replace hyphens with spaces, e.g. "tjue-en" -> "tjue en"
    except Exception as e:
        # print(f"Warning: num2words failed for '{n_str}' with type '{to_type}': {e}")
        return n_str # Return original if conversion fails

def _normalize_numbers_no(text):
    # This is a simplified version. A more robust solution would handle more cases.
    # Order of regex matters.

    # Years (e.g., "år 1995", "i 2023") - needs to be handled carefully not to capture all 4-digit numbers
    text = re.sub(r'\b(\d{4})\b', lambda m: _norwegian_number_to_words(m.group(1), 'year'), text)

    # Ordinals (e.g., "3. etasje", "den 1.") - very basic for now
    # This regex is too simple and might misinterpret. `num2words` ordinal is better if applicable.
    # Example: "3. etg." -> (abbr expand) "3. etasje" -> (ordinal) "tredje etasje"
    # This specific case might need a combined rule or careful ordering.
    # For now, let's assume abbreviations are expanded first. "3. etasje"
    # text = re.sub(r'\b(\d+)\.(?=\s*(?:etasje|plass|gang|del)\b)', lambda m: _norwegian_number_to_words(m.group(1), 'ordinal'), text)
    # Awaiting better ordinal handling strategy or direct use of num2words for specific patterns.

    # Decimals (e.g., "1,23" -> "en komma to tre")
    text = re.sub(r'(\d+),(\d+)',
                  lambda m: f"{_norwegian_number_to_words(m.group(1))} komma {_norwegian_number_to_words(m.group(2))}",
                  text) # Simple version, assumes second part is read digit by digit or as whole.
                        # For "1,23" -> "en komma tjue tre" (if m.group(2) is treated as number)
                        # or "en komma to tre" (if m.group(2) is read digit by digit "2" "3")
                        # Let's assume it's read as a full number for now.

    # Currency (e.g., "100 kr", "kr 50,50") - basic
    text = re.sub(r'(\d+[\.,]?\d*)\s*kr\.?', lambda m: f"{_norwegian_number_to_words(m.group(1).replace(',', '.'))} kroner", text) # kr after
    text = re.sub(r'kr\.?\s*(\d+[\.,]?\d*)', lambda m: f"{_norwegian_number_to_words(m.group(1).replace(',', '.'))} kroner", text) # kr before

    # General integers
    text = re.sub(r'\b\d+\b', lambda m: _norwegian_number_to_words(m.group(0)), text)

    return text

# General text cleaning rules
def _clean_text_no(text):
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Standardize common punctuation (example)
    text = text.replace('–', '-').replace('—', '-') # en/em dash to hyphen
    text = text.replace('“', '"').replace('”', '"')
    text = text.replace('‘', "'").replace('’', "'")
    # Add other cleaning rules as needed
    return text

def text_normalize(text):
    text = text.lower()
    text = _clean_text_no(text) # Apply general cleaning first
    text = _expand_abbreviations_no(text)
    text = _normalize_numbers_no(text) # This needs to be robust
    # Ensure final cleanup of whitespace that might be introduced by replacements
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- Grapheme-to-Phoneme (g2p) ---

MODEL_ID_NO = 'NbAiLab/nb-bert-base'
try:
    tokenizer_no = AutoTokenizer.from_pretrained(MODEL_ID_NO)
except Exception as e:
    print(f"Warning: Could not load Norwegian tokenizer {MODEL_ID_NO}. Using bert-base-multilingual-cased. Error: {e}")
    MODEL_ID_NO = 'bert-base-multilingual-cased'
    tokenizer_no = AutoTokenizer.from_pretrained(MODEL_ID_NO)

def _phonemize_espeak_no(text):
    """
    Uses espeak-ng to phonemize text for Norwegian.
    Returns a list of IPA phoneme characters.
    """
    try:
        # Replace hyphens as they can cause issues with espeak word interpretation
        # Also ensure text is properly handled if it contains quotes or special shell characters,
        # though subprocess list args should handle this.
        word_to_phonemize = text.replace('-', ' ')

        process = subprocess.run(
            ['espeak-ng', '-q', '-v', 'no', '--ipa=3', word_to_phonemize],
            capture_output=True, text=True, check=True, encoding='utf-8'
        )
        phoneme_string = process.stdout.strip()
        # Remove Zero Width Non-Joiner characters if espeak inserts them
        phoneme_string = phoneme_string.replace('\u200C', '')
        # Each character in the IPA string is considered a phoneme or a modifier (stress, length)
        return list(phoneme_string)
    except FileNotFoundError:
        print("ERROR: espeak-ng not found. Please install it.")
        return [symbols.UNK] # Fallback: return a list with a single UNK symbol
    except subprocess.CalledProcessError as e:
        print(f"ERROR: espeak-ng failed for text '{text}' (processed as '{word_to_phonemize}'): {e.stderr}")
        return [symbols.UNK] # Fallback
    except Exception as e:
        print(f"An unexpected error occurred during phonemization for '{text}': {e}")
        return [symbols.UNK] # Fallback

def distribute_phone(n_phone, n_word):
    phones_per_word = [0] * n_word
    if n_word == 0:
        return []
    for i in range(n_phone):
        phones_per_word[i % n_word] += 1
    return phones_per_word

def g2p(text, pad_start_end=True, tokenized_text=None):
    normalized_text = text_normalize(text)

    if tokenized_text is None:
        tokenized_text = tokenizer_no.tokenize(normalized_text)

    phones = []
    tones = []
    word2ph = []

    ph_groups = []
    for t in tokenized_text:
        if not t.startswith("##") and not (len(ph_groups)>0 and ph_groups[-1][-1].endswith("-") ):
             ph_groups.append([t])
        else:
            if t.startswith("##"):
                ph_groups[-1].append(t[2:])
            else:
                ph_groups[-1].append(t)

    for group in ph_groups:
        word = "".join(group)

        current_phonemes = []
        if word in tokenizer_no.all_special_tokens:
            if word == tokenizer_no.unk_token:
                current_phonemes = [symbols.UNK]
            else:
                current_phonemes = [] # Or symbols.SP for silence?
        else:
            # Clean the word for espeak: e.g. remove punctuation that espeak might misinterpret
            # or that the tokenizer preserved. This step depends on tokenizer behavior.
            # For now, assume `word` is relatively clean (mostly alphabetic).
            # The `_phonemize_espeak_no` handles internal hyphens.
            # Consider stripping common punctuation if tokenizer passes it through.
            # word_for_espeak = re.sub(r"[^\w\s'-]", "", word) # Example: keep alphanumeric, space, hyphen, apostrophe
            word_for_espeak = word # Using word directly for now, _phonemize_espeak_no handles hyphens.

            if not word_for_espeak.strip(): # If word is only punctuation/whitespace after cleaning
                 current_phonemes = [symbols.SP] if word_for_espeak else [] # SP for space-like, empty otherwise
            else:
                current_phonemes = _phonemize_espeak_no(word_for_espeak)

        current_phonemes = [p for p in current_phonemes if p] # Filter out empty strings

        # Map phonemes to our symbol list, or UNK
        mapped_phonemes = []
        for p in current_phonemes:
            if p in symbols.symbols:
                mapped_phonemes.append(p)
            # elif p.lower() in symbols.symbols: # Optional: try lowercasing if symbol not found
            #     mapped_phonemes.append(p.lower())
            else:
                mapped_phonemes.append(symbols.UNK)
        current_phonemes = mapped_phonemes

        if not current_phonemes and word.strip() and word not in tokenizer_no.all_special_tokens:
             # Fallback for non-special words if espeak returns nothing
            current_phonemes = [symbols.UNK] * len(word)


        phones.extend(current_phonemes)
        tones.extend([0] * len(current_phonemes))

        phones_for_this_word_group = distribute_phone(len(current_phonemes), len(group))
        word2ph.extend(phones_for_this_word_group)

    if pad_start_end:
        phones = [symbols.pad] + phones + [symbols.pad]
        tones = [0] + tones + [0]
        word2ph = [1] + word2ph + [1]

    return phones, tones, word2ph

# --- BERT Feature Extraction ---
def get_bert_feature(text, word2ph, device=None):
    try:
        from . import norwegian_bert
        return norwegian_bert.get_bert_feature(text, word2ph, tokenizer_no, MODEL_ID_NO, device=device)
    except ImportError:
        # print("Warning: melo.text.norwegian_bert not found. BERT features will not be generated for Norwegian.")
        raise NotImplementedError("Norwegian BERT feature extraction is not fully implemented yet. Requires melo.text.norwegian_bert.")
    except Exception as e:
        # print(f"Error during Norwegian BERT feature extraction: {e}")
        raise NotImplementedError(f"Norwegian BERT feature extraction failed: {e}")


if __name__ == '__main__':
    print("Testing Norwegian Text Processing...")
    sample_text = "Dette er en test på norsk, med tall 123 og forkortelser som f.eks."
    print(f"Original: {sample_text}")
    normalized = text_normalize(sample_text)
    print(f"Normalized: {normalized}")

    print("\nTesting G2P (ensure espeak-ng is installed and 'no' voice is available):")
    sample_g2p_text = "Flyndre"
    print(f"G2P for: '{sample_g2p_text}'")

    phones, tones, word2ph = g2p(sample_g2p_text, pad_start_end=True)
    print(f"Phones: {phones}")
    print(f"Tones: {tones}")
    print(f"Word2Ph: {word2ph}")

    norwegian_chars_text = "blåbærsyltetøy på ei øy"
    print(f"\nG2P for: '{norwegian_chars_text}'")
    phones, tones, word2ph = g2p(norwegian_chars_text)
    print(f"Phones: {phones}")
    print(f"Tones: {tones}")
    print(f"Word2Ph: {word2ph}")

    longer_sentence = "Jeg liker å programmere datasystemer, spesielt for talesyntese."
    print(f"\nG2P for: '{longer_sentence}'")
    phones, tones, word2ph = g2p(longer_sentence)
    print(f"Phones: {phones}")

    unknown_word_text = "Xyzzyq"
    print(f"\nG2P for unknown word: '{unknown_word_text}'")
    phones, tones, word2ph = g2p(unknown_word_text)
    print(f"Phones: {phones}")

    print("\nDone with norwegian.py tests.")
    # To run: python -m melo.text.norwegian
    # Ensure espeak-ng is installed and Norwegian voice data is available.
    # The actual phonemes in `no_symbols` in `symbols.py` should be verified against espeak-ng's output.
    # Example: espeak-ng -v no --ipa=3 -q "Hallo"
    # Output might be like: hˈɑluː (check actual output)
    # Then ensure 'h', 'ˈ', 'ɑ', 'l', 'u', 'ː' are in no_symbols.
    # The current parsing `list(phoneme_string)` assumes each char is a symbol.
    # If espeak outputs multi-char symbols not intended to be split (e.g., specific affricate symbols),
    # this needs more advanced parsing or those multi-char symbols added to no_symbols.
    # For standard IPA, `list()` is usually a good start.
    # Check for encoding issues if strange characters appear. `encoding='utf-8'` in subprocess.run is good.
    # The `word_for_espeak = word.replace('-', ' ')` in `_phonemize_espeak_no` helps with words like "well-being".
    # Consider other word cleaning in `g2p` before calling `_phonemize_espeak_no` if tokenizer passes much punctuation.
    # The mapping to `symbols.UNK` if `p not in symbols.symbols` is critical.
    # If `no_symbols` is comprehensive, UNK should only appear for truly unhandled espeak outputs.
    # The fallback `[symbols.UNK] * len(word)` in `g2p` is a last resort.
    # `symbols.SP` (space) could be used for punctuation or empty tokens if desired.
    # Current code maps empty/special tokens (not UNK) to empty phoneme lists, which are then ignored.
    # This might be fine, or they could map to SP.
    # Example: `g2p("Hello , world")` -> tokenizer might give `['Hello', ',', 'world']`.
    # If `,` is tokenized, current code might make it an empty phoneme list.
    # This behavior needs to be consistent with how other languages are processed in the main model.
    # The `NotImplementedError` for BERT is expected for now.
    print("\nAttempting to get espeak-ng version for diagnostics:")
    try:
        version_proc = subprocess.run(['espeak-ng', '--version'], capture_output=True, text=True, check=True, encoding='utf-8')
        print(f"espeak-ng version: {version_proc.stdout.strip()}")
    except Exception as e:
        print(f"Could not get espeak-ng version: {e}")
