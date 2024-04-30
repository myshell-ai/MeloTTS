import pytest
import torch
from melo.text.thai import (
    normalize,
    word_tokenize,
    thai_text_to_phonemes,
    text_normalize,
    g2p,
    get_bert_feature,
)
from melo.text.korean import (
    text_normalize as k_text_normalize,
    get_bert_feature as k_get_bert_feature,
    g2p as k_g2p,
)

def test_normalize():
    text = "   ข้อความ ภาษา ไทย 123 ABC   "
    normalized_text = normalize(text)
    assert normalized_text == "ข้อความ ภาษา ไทย หนึ่งร้อยยี่สิบสาม เอบีซี"

def test_word_tokenize():
    text = "ฉันเข้าใจคุณค่าของงานของฉันและความหมายของสิ่งที่ฟอนเทนทำเพื่อคนทั่วไปเป็นอย่างดี"
    tokenized_text = word_tokenize(text, engine="newmm")
    assert tokenized_text == ['ฉัน', 'เข้าใจ', 'คุณค่า', 'ของ', 'งาน', 'ของ', 'ฉัน', 'และ', 'ความหมาย', 'ของ', 'สิ่ง', 'ที่', 'ฟอน', 'เท', 'น', 'ทำ', 'เพื่อ', 'คน', 'ทั่วไป', 'เป็น', 'อย่าง', 'ดี']

def test_thai_text_to_phonemes():
    text = "สวัสดีครับ"
    phonemes = thai_text_to_phonemes(text)
    assert phonemes == "s a ˨˩ . w a t̚ ˨˩ . d iː ˧ kʰ r a p̚ ˦˥"

def test_g2p():
    text = "ฉันรักเมืองไทย"
    normalized_text = text_normalize(text)
    phones, tones, word2ph = g2p(normalized_text)
    assert phones == ['_', 't͡ɕʰ', 'a', 'n', '˩˩˦', 'r', 'a', 'k̚', '˦˥', 'm', 'ɯa̯', 'ŋ', '˧', 'tʰ', 'aj', '˧', '.', 'j', 'a', '˦˥', '.', '_']
    assert tones == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    assert word2ph == [1, 7, 7, 6, 1]

def test_get_bert_feature():
    text = "ฉันเข้าใจคุณค่าของงานของฉันและความหมายของสิ่งที่ฟอนเทนทำเพื่อคนทั่วไปเป็นอย่างดี"
    normalized_text = text_normalize(text)
    phones, tones, word2ph = g2p(normalized_text)

    bert_features = get_bert_feature(normalized_text, word2ph, device='cpu')

    assert isinstance(bert_features, torch.Tensor), "bert_features should be a torch.Tensor"
    assert bert_features.shape[0] == 768, f"Expected bert_features.shape[0] to be 768, but got {bert_features.shape[0]}"

    # Modify the assertion to check the number of phones instead of the length of word2ph
    num_phones = sum(word2ph)
    assert bert_features.shape[1] == num_phones, f"Expected bert_features.shape[1] to be {num_phones}, but got {bert_features.shape[1]}"

    # Additional assertions to check the values of bert_features
    assert not torch.isnan(bert_features).any(), "bert_features should not contain any NaN values"
    assert not torch.isinf(bert_features).any(), "bert_features should not contain any infinity values"

# Compare with Korean
def test_get_bert_feature_korean():
    text = "저는 제 일의 가치와 의미를 잘 알고 있습니다. 앞으로도 저는 제 일에 자부심을 갖고 살아갈 것입니다."
    normalized_text = k_text_normalize(text)
    phones, tones, word2ph = k_g2p(normalized_text)

    bert_features = k_get_bert_feature(normalized_text, word2ph, device='cpu')

    assert isinstance(bert_features, torch.Tensor), "bert_features should be a torch.Tensor"
    assert bert_features.shape[0] == 768, f"Expected bert_features.shape[0] to be 768, but got {bert_features.shape[0]}"

    # Modify the assertion to check the number of phones instead of the length of word2ph
    num_phones = sum(word2ph)
    assert bert_features.shape[1] == num_phones, f"Expected bert_features.shape[1] to be {num_phones}, but got {bert_features.shape[1]}"

    # Additional assertions to check the values of bert_features
    assert not torch.isnan(bert_features).any(), "bert_features should not contain any NaN values"
    assert not torch.isinf(bert_features).any(), "bert_features should not contain any infinity values"
