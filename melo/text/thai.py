import re
import unicodedata
from transformers import AutoTokenizer
from . import punctuation, symbols
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
    return thai_g2p_dict.get(word, list(word))

def thai_text_to_phonemes(text):
    text = normalize(text)
    words = word_tokenize(text, engine="newmm")
    phonemes = []
    for word in words:
        word_phonemes = map_word_to_phonemes(word)
        phonemes.extend(word_phonemes)
    return " ".join(phonemes)

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

def g2p(norm_text):
    tokenized = tokenizer.tokenize(norm_text)
    phs = []
    ph_groups = []
    current_group = []  # Track the current group of tokens

    for t in tokenized:
        if t.startswith("▁"):  # Start of a new word or phrase
            if current_group:  # Append current group to ph_groups if not empty
                ph_groups.append(current_group)
                current_group = []  # Reset current_group for the new word or phrase
        current_group.append(t.replace("▁", ""))  # Add token to current_group

    if current_group:  # Append the last group if not empty
        ph_groups.append(current_group)

    word2ph = []

    for group in ph_groups:
        text = "".join(group)  # Concatenate tokens in the group to form the word or phrase
        if text == '[UNK]': # handle special cases like unknown tokens ("[UNK]")
            phs.append('_')
            word2ph.append(1)
            continue
        elif text in punctuation:
            phs.append(text)
            word2ph.append(1)
            continue
        phonemes = thai_text_to_phonemes(text)
        phone_len = len(phonemes.split())
        word_len = len(group)
        aaa = distribute_phone(phone_len, word_len)
        assert len(aaa) == word_len
        word2ph.extend(aaa)
        phs.extend(phonemes.split())

    phones = ["_"] + phs + ["_"]
    tones = [0 for _ in phones]
    word2ph = [1] + word2ph + [1]
    assert len(word2ph) == len(tokenized) + 2

    return phones, tones, word2ph

def get_bert_feature(text, word2ph, device='cuda', model_id='airesearch/wangchanberta-base-att-spm-uncased'):
    from . import thai_bert
    return thai_bert.get_bert_feature(text, word2ph, device=device, model_id=model_id)

if __name__ == "__main__":
    from text.symbols import symbols
    text = "ฉันเข้าใจคุณค่าของงานของฉันและความหมายของสิ่งที่ฟอนเทนทำเพื่อคนทั่วไปเป็นอย่างดี ฉันจะใช้ชีวิตอย่างภาคภูมิใจในงานของฉันต่อไป"
    import json

    # Load Thai dataset
    thai_data = json.load(open('thai_dataset.json'))
    from tqdm import tqdm
    new_symbols = []
    for key, item in tqdm(thai_data.items()):
        texts = item.get('voiceContent', '')
        if isinstance(texts, list):
            texts = ','.join(texts)
        if texts is None:
            continue
        if len(texts) == 0:
            continue

        text = text_normalize(text)
        phones, tones, word2ph = g2p(text)
        bert = get_bert_feature(text, word2ph, device='cuda', model_id=model_id)

        for ph in phones:
            if ph not in symbols and ph not in new_symbols:
                new_symbols.append(ph)
                print('update!, now symbols:')
                print(new_symbols)
                with open('thai_symbol.txt', 'w') as f:
                    f.write(f'{new_symbols}')
