import re
from transformers import AutoTokenizer
from . import symbols

def distribute_phone(n_phone, n_word):
    phones_per_word = [0] * n_word
    for task in range(n_phone):
        min_tasks = min(phones_per_word)
        min_index = phones_per_word.index(min_tasks)
        phones_per_word[min_index] += 1
    return phones_per_word

def text_normalize(text):
    # Basic Turkish text normalization
    # Convert to lowercase while preserving Turkish characters
    text = text.replace("I", "ı").lower()
    text = text.lower()
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)
    # Remove unnecessary punctuation
    text = re.sub(r'[^\w\s.,!?;:əğıöüşçİĞÖÜŞÇ]', '', text)
    return text.strip()

def post_replace_ph(ph):
    rep_map = {
        "：": ",",
        "；": ",",
        "，": ",",
        "。": ".",
        "！": "!",
        "？": "?",
        "\n": ".",
        "·": ",",
        "、": ",",
        "...": "…"
    }
    if ph in rep_map.keys():
        ph = rep_map[ph]
    if ph in symbols:
        return ph
    if ph not in symbols:
        ph = "UNK"
    return ph

def refine_ph(phn):
    tone = 0
    if re.search(r"\d$", phn):
        tone = int(phn[-1]) + 1
        phn = phn[:-1]
    return phn.lower(), tone

def tr_to_ipa(text):
    """
    Convert Turkish text to IPA
    This is a basic implementation - you might want to expand this based on Turkish phonology rules
    """
    tr_to_ipa_dict = {
        'a': 'a', 'e': 'e', 'ı': 'ɯ', 'i': 'i', 
        'o': 'o', 'ö': 'ø', 'u': 'u', 'ü': 'y',
        'b': 'b', 'c': 'dʒ', 'ç': 'tʃ', 'd': 'd',
        'f': 'f', 'g': 'ɡ', 'ğ': 'ː', 'h': 'h',
        'j': 'ʒ', 'k': 'k', 'l': 'l', 'm': 'm',
        'n': 'n', 'p': 'p', 'r': 'r', 's': 's',
        'ş': 'ʃ', 't': 't', 'v': 'v', 'y': 'j',
        'z': 'z'
    }
    return ''.join(tr_to_ipa_dict.get(char, char) for char in text.lower())

# Initialize the Turkish BERT tokenizer
model_id = 'ytu-ce-cosmos/turkish-base-bert-uncased.'
tokenizer = AutoTokenizer.from_pretrained(model_id)

def g2p(text, pad_start_end=True, tokenized=None):
    if tokenized is None:
        tokenized = tokenizer.tokenize(text)
    
    phs = []
    ph_groups = []
    for t in tokenized:
        if not t.startswith("##"):  # Note: Turkish BERT uses ## for subwords
            ph_groups.append([t])
        else:
            ph_groups[-1].append(t.replace("##", ""))
    
    phones = []
    tones = []
    word2ph = []
    
    for group in ph_groups:
        w = "".join(group)
        phone_len = 0
        word_len = len(group)
        if w == '[UNK]':
            phone_list = ['UNK']
        else:
            phone_list = list(filter(lambda p: p != " ", tr_to_ipa(w)))
        
        for ph in phone_list:
            phones.append(ph)
            tones.append(0)  # Turkish is not a tonal language
            phone_len += 1
        aaa = distribute_phone(phone_len, word_len)
        word2ph += aaa

    if pad_start_end:
        phones = ["_"] + phones + ["_"]
        tones = [0] + tones + [0]
        word2ph = [1] + word2ph + [1]
    return phones, tones, word2ph

def get_bert_feature(text, word2ph, device=None):
    from text import turkish_bert
    return turkish_bert.get_bert_feature(text, word2ph, device=device)

if __name__ == "__main__":
    text = "Merhaba, nasılsın? Ben iyiyim."
    text = text_normalize(text)
    print(text)
    phones, tones, word2ph = g2p(text)
    bert = get_bert_feature(text, word2ph)
    print(phones)
    print(len(phones), tones, sum(word2ph), bert.shape)
