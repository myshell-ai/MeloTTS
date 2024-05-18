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
thai_g2p_dict = {}

with open("wiktionary-23-7-2022-clean.tsv", "r", encoding="utf-8") as f:
    for line in f:
        word, phonemes = line.strip().split("\t")
        thai_g2p_dict[word] = phonemes.split()

def map_word_to_phonemes(word):
    print(f"Mapping word to phonemes: {word}")
    phonemes = thai_g2p_dict.get(word, list(word))
    if not phonemes:
        phonemes = list(word)
    print(f"Phonemes for the word: {phonemes}")
    return " ".join(phonemes)

def thai_text_to_phonemes(text):
    print(f"Original text: {text}")
    text = normalize(text)
    print(f"Normalized text: {text}")
    words = word_tokenize(text, engine="newmm")
    print(f"Tokenized words: {words}")
    phonemes = []
    for word in words:
        print(f"Processing word: {word}")
        word_phonemes = map_word_to_phonemes(word)
        print(f"Word phonemes: {word_phonemes}")
        phonemes.append(word_phonemes)
    print(f"Final phonemes: {phonemes}")
    return " . ".join(phonemes)

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

model_id = 'airesearch/wangchanberta-base-att-spm-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_id)

tone_map = {
    "˧": 2,  # Mid tone
    "˨˩": 1,  # Low tone
    "˦˥": 3,  # Rising tone
    "˩˩˦": 4,  # Falling tone
    "˥˩": 5,  # High tone
}

def extract_tones(phs):
    tones = []
    for ph in phs:
        if ph in tone_map:
            tones = [tone_map[ph]] * (len(phs) - 1)
            break
    if not tones:
        tones = [0] * len(phs)
    return tones

def g2p(norm_text):
    print(f"Normalized text: {norm_text}")
    words = word_tokenize(norm_text, engine="newmm")
    print(f"Tokenized words: {words}")
    phonemes_and_tones = []

    for word in words:
        print(f"Processing word: {word}")
        phonemes = thai_text_to_phonemes(word)
        print(f"Phonemes for the word: {phonemes}")
        phoneme_groups = phonemes.split(".")
        print(f"Phoneme groups: {phoneme_groups}")
        word_phonemes = []
        word_tones = []

        for group in phoneme_groups:
            print(f"Processing phoneme group: {group}")
            group_phonemes = [ph for ph in group.split() if ph not in tone_map]
            group_tones = extract_tones(group.split())
            print(f"Group phonemes: {group_phonemes}")
            print(f"Group tones: {group_tones}")
            word_phonemes.extend(group_phonemes)
            word_tones.extend(group_tones)

        print(f"Word phonemes: {word_phonemes}")
        print(f"Word tones: {word_tones}")
        phonemes_and_tones.append((word_phonemes, word_tones))

    joined_text = " ".join(words)
    print(f"Joined text: {joined_text}")
    bert_tokens = tokenizer.tokenize(joined_text)
    print(f"BERT tokens: {bert_tokens}")

    phs = []
    tones = []
    word2ph = []

    for word, (word_phonemes, word_tones) in zip(words, phonemes_and_tones):
        print(f"Aligning word: {word}")
        print(f"Word phonemes: {word_phonemes}")
        print(f"Word tones: {word_tones}")
        phs.extend(word_phonemes)
        tones.extend(word_tones)
        word2ph.append(len(word_phonemes))

    phs = ["_"] + phs + ["_"]
    tones = [0] + tones + [0]
    word2ph = [1] + word2ph + [1]

    print(f"Final phs: {phs}")
    print(f"Final tones: {tones}")
    print(f"Final word2ph: {word2ph}")

    assert len(word2ph) == len(words) + 2
    return phs, tones, word2ph

def get_bert_feature(text, word2ph, device='cuda', model_id='airesearch/wangchanberta-base-att-spm-uncased'):
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
