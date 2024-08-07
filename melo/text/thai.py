import re
import unicodedata
from transformers import AutoTokenizer
from . import punctuation, symbols, pu_symbols
from num2words import num2words
from pythainlp.tokenize import word_tokenize
from pythainlp.transliterate import romanize
from pythainlp.util import normalize as thai_normalize
from pythainlp.util import thai_to_eng, eng_to_thai
from melo.text.thai_dictionary import english_dictionary, etc_dictionary
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def normalize_with_dictionary(text, dic):
    if any(key in text for key in dic.keys()):
        pattern = re.compile("|".join(re.escape(key) for key in dic.keys()))
        return pattern.sub(lambda x: dic[x.group()], text)
    return text

def normalize(text):
    text = text.strip()
    text = thai_normalize(text)
    text = normalize_with_dictionary(text, etc_dictionary)
    text = re.sub(r"\d+", lambda x: num2words(int(x.group()), lang="th"), text)
    text = normalize_english(text)
    text = text.lower()
    return text

def normalize_english(text):
    def fn(m):
        word = m.group()
        if word.upper() in english_dictionary:
            return english_dictionary[word.upper()]
        return "".join(english_dictionary.get(char.upper(), char) for char in word)

    text = re.sub(r"([A-Za-z]+)", fn, text)
    return text

# Load the Thai G2P dictionary
thai_g2p_dict = defaultdict(list)

with open("wiktionary-23-7-2022-clean.tsv", "r", encoding="utf-8") as f:
    for line in f:
        word, phonemes = line.strip().split("\t")
        thai_g2p_dict[word].append(phonemes.split())
        thai_g2p_dict["ะ"] = ["a"]

def map_word_to_phonemes(word):
    logger.debug(f"Looking up word: {word}")

    # First, try to find the whole word in the dictionary
    phonemes_list = thai_g2p_dict.get(word)
    if phonemes_list:
        logger.debug(f"Found whole word {word} in dictionary")
        return " ".join(phonemes_list[0])

    # If not found, try to split the word
    subwords = word_tokenize(word, engine="newmm")

    if len(subwords) > 1:
        logger.debug(f"Word {word} split into subwords: {subwords}")
        # If the word can be split, recursively process each subword
        return " . ".join(map_word_to_phonemes(subword) for subword in subwords)
    else:
        logger.debug(f"Word {word} cannot be split, processing character by character")
        return map_partial_word(word)

def map_partial_word(word):
    if not word:
        return ""

    logger.debug(f"Mapping partial word: {word}")

    # Handle Thanthakhat (์) character
    if len(word) > 1 and word[1] == '์':
        logger.debug(f"Found Thanthakhat, skipping {word[:2]}")
        return map_partial_word(word[2:])

    # Handle vowels and tone marks
    if word[0] in thai_vowels or word[0] in thai_tone_marks:
        phoneme = thai_g2p_dict.get(word[0], [word[0]])[0]
        return phoneme + " " + map_partial_word(word[1:])

    # Try to find the longest matching prefix
    for i in range(len(word), 0, -1):
        prefix = word[:i]
        phonemes_list = thai_g2p_dict.get(prefix)
        if phonemes_list:
            logger.debug(f"Found matching prefix: {prefix}")
            return " ".join(phonemes_list[0]) + " " + map_partial_word(word[i:])

    # If no match found, return the first character and continue with the rest
    logger.debug(f"No match found for {word[0]}, continuing with rest")
    return word[0] + " " + map_partial_word(word[1:])

# Comprehensive mapping of Thai characters to their phonetic representations
thai_char_to_phoneme = {
    # Consonants
    'ก': 'k',
    'ข': 'kʰ',
    'ฃ': 'kʰ',
    'ค': 'kʰ',
    'ฅ': 'kʰ',
    'ฆ': 'kʰ',
    'ง': 'ŋ',
    'จ': 't͡ɕ',
    'ฉ': 't͡ɕʰ',
    'ช': 't͡ɕʰ',
    'ซ': 's',
    'ฌ': 't͡ɕʰ',
    'ญ': 'j',
    'ฎ': 'd',
    'ฏ': 't',
    'ฐ': 'tʰ',
    'ฑ': 'tʰ',
    'ฒ': 'tʰ',
    'ณ': 'n',
    'ด': 'd',
    'ต': 't',
    'ถ': 'tʰ',
    'ท': 'tʰ',
    'ธ': 'tʰ',
    'น': 'n',
    'บ': 'b',
    'ป': 'p',
    'ผ': 'pʰ',
    'ฝ': 'f',
    'พ': 'pʰ',
    'ฟ': 'f',
    'ภ': 'pʰ',
    'ม': 'm',
    'ย': 'j',
    'ร': 'r',
    'ล': 'l',
    'ว': 'w',
    'ศ': 's',
    'ษ': 's',
    'ส': 's',
    'ห': 'h',
    'ฬ': 'l',
    'อ': 'ʔ',
    'ฮ': 'h',

    # Vowels
    'ะ': 'a',
    'ั': 'a',
    'า': 'aː',
    'ำ': 'am',
    'ิ': 'i',
    'ี': 'iː',
    'ึ': 'ɯ',
    'ื': 'ɯː',
    'ุ': 'u',
    'ู': 'uː',
    'เ': 'eː',
    'แ': 'ɛː',
    'โ': 'oː',
    'ใ': 'aj',
    'ไ': 'aj',
    '็': '',  # Short vowel marker
    'ๆ': '',  # Repetition marker

    # Tone marks
    '่': '˨˩',  # Low tone
    '้': '˦˥',  # Rising tone
    '๊': '˥˩',  # Falling tone
    '๋': '˧',   # High tone

    # Special characters
    '์': '',  # Thanthakhat (cancels sound of preceding consonant)
}

def map_remaining_thai_chars(phones):
    mapped_phones = []
    for phone in phones:
        if phone in thai_char_to_phoneme:
            mapped_phones.append(thai_char_to_phoneme[phone])
        else:
            mapped_phones.append(phone)
    return mapped_phones

def thai_text_to_phonemes(text):
    text = normalize(text)
    words = word_tokenize(text, engine="newmm")
    logger.debug(f"word_tokenize output: {words}")
    phonemes = []
    for word in words:
        word_phonemes = map_word_to_phonemes(word)
        phonemes.extend(word_phonemes.split())

    # Map any remaining Thai characters
    mapped_phonemes = map_remaining_thai_chars(phonemes)

    return " ".join(mapped_phonemes)

# Define Thai vowels, tone marks, and special characters
thai_vowels = set("ะัาำิีึืุูเแโใไฤฦ็")
thai_tone_marks = set("่้๊๋")
thai_special_chars = set("์ๆฯ๎๏") # Thanthakhat, Maiyamok, Paitai, and Phinthu

# Update the thai_g2p_dict with proper mappings for vowels and tone marks
thai_g2p_dict.update({
    'โ': ['o'],
    'ใ': ['aj'],
    'ไ': ['aj'],
    'แ': ['ɛː'],
    'เ': ['eː'],
    'ฤ': ['rɯ'],
    'ฦ': ['lɯ'],
    '็': ['ː'], # Mai Taikhu (used to shorten a vowel)
    '่': ['˨˩'], # Low tone
    '้': ['˦˥'], # Rising tone
    '๊': ['˥˩'], # Falling tone
    '๋': ['˧'], # High tone
    '์': [''], # Thanthakhat (cancels the sound of the syllable)
    'ๆ': [''], # Maiyamok (repetition mark)
    'ฯ': [''], # Paitai (abbreviation mark)
    '๎': [''], # Phinthu (used to indicate a silent consonant)
    '๏': [''], # Angkhankhu (used to mark the end of a paragraph or section)
})


def text_normalize(text):
    text = normalize(text)
    return text

def distribute_phone(n_phone, n_word):
    phones_per_word = [0] * n_word
    for task in range(n_phone):
        min_tasks = min(phones_per_word)
        min_index = phones_per_word.index(min_tasks)
        phones_per_word[min_index] += 1
    return phones_per_word

model_id = 'clicknext/phayathaibert'
tokenizer = AutoTokenizer.from_pretrained(model_id)

tone_map = {
    "˧": 2,  # Mid tone
    "˨˩": 1,  # Low tone
    "˦˥": 3,  # Rising tone
    "˩˩˦": 4,  # Falling tone
    "˥˩": 5,  # High tone
}


# def extract_tones(phs):
#     tones = []
#     tone_value = 2  # Default tone value when no tone symbol is found
#     last_item = phs[-1]

#     for ph in phs:
#         if ph in tone_map:
#             tone_value = tone_map[ph]

#     # Assign the tone value to each phoneme excluding the tone marker if there is one
#     if last_item in tone_map:
#         tones = [tone_value] * (len(phs) - 1)
#     else:
#         tones = [tone_value] * (len(phs))

#     print("=======> PHS: ",  phs)
#     print("=======> TONES: ",  tones)

#     return tones

def extract_tones(phs):
    phonemes = []
    tones = []
    current_tone = 2  # Default mid tone
    for ph in phs.split():
        if ph in tone_map:
            current_tone = tone_map[ph]
        elif ph == '_':
            phonemes.append(ph)
            tones.append(0)  # Zero tone for underscore
        else:
            phonemes.append(ph)
            tones.append(current_tone)
    return phonemes, tones

# def g2p_bert(norm_text, pad_start_end=True):
#     tokenized = tokenizer.tokenize(norm_text)
#     print("tokenized text")
#     phs = []
#     tones = []
#     word2ph = []

#     ph_groups = []
#     for t in tokenized:
#         if t.startswith("▁"):
#             ph_groups.append([t])
#         else:
#             ph_groups[-1].append(t)

#     for group in ph_groups:
#         word = "".join(group).replace("▁", "")
#         phonemes = thai_text_to_phonemes(word)
#         phoneme_groups = phonemes.split(".")
#         phoneme_groups = list(filter(str.strip, phoneme_groups))

#         word_phonemes = []
#         word_tones = []
#         word_phones_count = []

#         for p_group in phoneme_groups:
#             group_phonemes = [ph for ph in p_group.split() if ph not in tone_map]
#             group_tones = extract_tones(p_group.split())

#             word_phonemes.extend(group_phonemes)
#             word_tones.extend(group_tones)
#             word_phones_count.append(len(group_phonemes))

#         phs.extend(word_phonemes)
#         tones.extend(word_tones)
#         word2ph.extend(word_phones_count)

#     if pad_start_end:
#         phs = ["_"] + phs + ["_"]
#         tones = [1] + tones + [1]
#         word2ph = [1] + word2ph + [1]

#     print(f"Final phs: {phs}")
#     print(f"Final tones: {tones}")
#     print(f"Final word2ph: {word2ph}")

#     assert len(word2ph) == len(tokenized) + 2

#     return phs, tones, word2ph


def g2p_og(norm_text, pad_start_end=True):
    tokenized = tokenizer.tokenize(norm_text)
    phs = []
    tones = []
    word2ph = []

    print("tokenized", tokenized)

    for word in tokenized:
        # if word == "▁":
        #     word2ph.append(1)
        #     continue

        if word.startswith("▁"):
            word = word.replace("▁", "")

        phonemes = thai_text_to_phonemes(word)
        phoneme_groups = phonemes.split(".")
        # Keep only non-empty groups for cases with a trailing dot
        # i.e 'b ɤː ˧ . tʰ oː ˧ . r a ˦˥ .'
        phoneme_groups = [group for group in phoneme_groups if group.strip()]

        word_phonemes = []
        word_tones = []

        for group in phoneme_groups:
            group_phonemes = [ph for ph in group.split() if ph not in tone_map]
            group_tones = extract_tones(group.split())
            word_phonemes.extend(group_phonemes)
            word_tones.extend(group_tones)

        phs.extend(word_phonemes)
        tones.extend(word_tones)
        word2ph.append(len(word_phonemes))

    if pad_start_end:
        phs = ["_"] + phs + ["_"]
        tones = [1] + tones + [1]
        word2ph = [1] + word2ph + [1]

    print(f"Final phs: {phs}")
    print(f"Final tones: {tones}")
    print(f"Final word2ph: {word2ph}")

    # assert len(word2ph) == len([t for t in tokenized if t != "▁"]) + 2

    return phs, tones, word2ph


def g2p_no_undersscores(norm_text, pad_start_end=True):
    tokenized = tokenizer.tokenize(norm_text)
    # print("The tokenized text", tokenized)
    phs = []
    tones = []
    ph_groups = []

    # Group tokens
    for t in tokenized:
        if not t.startswith("#"):
            ph_groups.append([t])
        else:
            ph_groups[-1].append(t.replace("#", ""))

    word2ph = []
    for group in ph_groups:
        text = "".join(group)

        # Special token handling
        if text == '[UNK]':
            phs += ['_']
            tones += [0]
            word2ph += [1]
            continue
        elif text in punctuation:
            phs += [text]
            tones += [0]
            word2ph += [1]
            continue

        # Phoneme conversion for grouped text
        phonemes = thai_text_to_phonemes(text)
        phoneme_groups = phonemes.split(".")
        phoneme_groups = list(filter(str.strip, phoneme_groups))

        word_phonemes = []
        word_tones = []
        for p_group in phoneme_groups:
            group_phonemes = [ph for ph in p_group.split() if ph not in tone_map]
            group_tones = extract_tones(p_group.split())
            word_phonemes.extend(group_phonemes)
            word_tones.extend(group_tones)

        phone_len = len(word_phonemes)
        word_len = len(group)

        aaa = distribute_phone(phone_len, word_len)
        assert len(aaa) == word_len
        word2ph += aaa

        phs += word_phonemes
        tones += word_tones

    if pad_start_end:
        phs = ["_"] + phs + ["_"]
        tones = [0] + tones + [0]
        word2ph = [1] + word2ph + [1]

    # print(f"Final phs: {phs}")
    # print(f"Final tones: {tones}")
    # print(f"Final word2ph: {word2ph}")

    assert len(word2ph) == len(tokenized) + 2
    return phs, tones, word2ph


def g2p(norm_text, pad_start_end=True):
    tokenized = tokenizer.tokenize(norm_text)
    phs = []
    tones = []
    ph_groups = []

    # Group tokens
    for t in tokenized:
        if t == '▁':
            ph_groups.append([t])  # Add '▁' as its own group
        else:
            if not ph_groups or ph_groups[-1] == ['▁']:
                ph_groups.append([t])
            else:
                ph_groups[-1].append(t)

    word2ph = []
    for group in ph_groups:
        text = "".join(group)
        if text == '▁':
            phs += ['_']
            tones += [0]  # Zero tone for underscore
            word2ph += [1]
            continue
        elif text == '[UNK]':
            phs += ['_']
            tones += [0]  # Zero tone for unknown token
            word2ph += [1]
            continue
        elif text in punctuation:
            phs += [text]
            tones += [0]  # Zero tone for punctuation
            word2ph += [1]
            continue

        # Phoneme conversion for grouped text
        phonemes = thai_text_to_phonemes(text)
        phoneme_groups = phonemes.split(".")
        phoneme_groups = list(filter(str.strip, phoneme_groups))
        
        word_phonemes = []
        word_tones = []
        for p_group in phoneme_groups:
            group_phonemes, group_tones = extract_tones(p_group)
            word_phonemes.extend(group_phonemes)
            word_tones.extend(group_tones)
        
        phone_len = len(word_phonemes)
        word_len = len(group)
        aaa = distribute_phone(phone_len, word_len)
        assert len(aaa) == word_len
        word2ph += aaa
        phs += word_phonemes
        tones += word_tones

    if pad_start_end:
        phs = ["_"] + phs + ["_"]
        tones = [0] + tones + [0]  # Zero tone for start/end padding
        word2ph = [1] + word2ph + [1]

    # assert len(phs) == len(tones)
    # assert len(phs) == sum(word2ph)
    
    return phs, tones, word2ph

def get_bert_feature(text, word2ph, device='cuda', model_id='clicknext/phayathaibert'):
    from . import thai_bert
    return thai_bert.get_bert_feature(text, word2ph, device=device, model_id=model_id)

if __name__ == "__main__":
    try:
        from text.symbols import symbols
        text = "ฉันเข้าใจคุณค่าของงานของฉันและความหมายของสิ่งที่ฟอนเทนทำเพื่อคนทั่วไปเป็นอย่างดี ฉันจะใช้ชีวิตอย่างภาคภูมิใจในงานของฉันต่อไป"

        text = text_normalize(text)
        phones, tones, word2ph = g2p(text)
        bert = get_bert_feature(text, word2ph, device='cuda', model_id=model_id)

        new_symbols = []
        for ph in phones:
            if ph not in symbols and ph not in new_symbols:
                new_symbols.append(ph)
                print('update!, now symbols:')
                print(new_symbols)
                with open('thai_symbol.txt', 'w') as f:
                    f.write(f'{new_symbols}')

    except Exception as e:
        print(f"An error occurred: {e}")
