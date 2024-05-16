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
    print(f"Phonemes for the word: {phonemes}")
    return phonemes

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
        phonemes.extend(word_phonemes)
    print(f"Final phonemes: {phonemes}")
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
    print(f"Normalized text: {norm_text}")
    tokenized = tokenizer.tokenize(norm_text)
    print(f"Tokenized text: {tokenized}")
    phs = []
    word2ph = []
    current_word = []
    current_phonemes = []

    for token in tokenized:
        print(f"Processing token: {token}")
        if token.startswith("▁"):
            print("Start of a new word")
            if current_word:
                word_phonemes = " ".join(current_phonemes)
                print(f"Word phonemes: {word_phonemes}")
                phs.extend(word_phonemes.split())
                word2ph.append(len(current_phonemes))
                current_word = []
                current_phonemes = []
            if token == "▁":
                phs.append("")
            else:
                current_word.append(token.replace("▁", ""))
                phonemes = thai_text_to_phonemes(token.replace("▁", ""))
                print(f"Phonemes: {phonemes}")
                current_phonemes.extend(phonemes.split())
        else:
            current_word.append(token)
            if token in punctuation or token in pu_symbols:
                print(f"Punctuation or symbol: {token}")
                phs.append(token)
                word2ph.append(1)
            else:
                phonemes = thai_text_to_phonemes(token.replace("▁", ""))
                print(f"Phonemes: {phonemes}")
                current_phonemes.extend(phonemes.split())

    if current_word:
        word_phonemes = " ".join(current_phonemes)
        print(f"Word phonemes: {word_phonemes}")
        phs.extend(word_phonemes.split())
        word2ph.append(len(current_phonemes))

    print(f"Final phs: {phs}")
    print(f"Final word2ph: {word2ph}")

    distributed_word2ph = []
    for i, group in enumerate(tokenized):
        if group.startswith("▁"):
            group = group.replace("▁", "")
            if group in punctuation or group in pu_symbols:
                distributed_word2ph.append(1)
            else:
                phonemes = thai_text_to_phonemes(group)
                distributed_word2ph.append(len(phonemes.split()))
        else:
            distributed_word2ph.append(1)  # Add 1 for spaces between words

    tone_markers = ['˥', '˦', '˧', '˨', '˩']
    phones = ["_"] + [re.sub(f'[{"".join(tone_markers)}]', '', p) for p in phs] + ["_"]
    print(f"Phones: {phones}")

    tones = extract_tones(phs)
    print(f"Tones: {tones}")

    word2ph = [1] + distributed_word2ph + [1]
    print(f"Final word2ph: {word2ph}")

    assert len(word2ph) == len(tokenized) + 2
    return phones, tones, word2ph

def extract_tones(phs):
    tones = []
    tone_map = {
        "˥": 5,  # High tone
        "˦": 4,  # Rising tone
        "˧": 3,  # Mid tone
        "˨": 2,  # Falling tone
        "˩": 1,  # Low tone
    }
    for ph in phs:
        tone_found = False
        for marker, value in tone_map.items():
            if marker in ph:
                tones.append(value)
                tone_found = True
                break
        if not tone_found:
            tones.append(0)
    return tones



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
