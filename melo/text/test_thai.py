import pytest
from melo.text.thai import normalize, tokenizer, thai_text_to_phonemes, text_normalize, g2p, get_bert_feature

def test_normalize():
    text = "   ข้อความ ภาษา ไทย 123 ABC   "
    normalized_text = normalize(text)
    assert normalized_text == "ข้อความ ภาษา ไทย หนึ่งร้อยยี่สิบสาม เอบีซี"

def test_tokenize():
    text = "ฉันเข้าใจคุณค่าของงานของฉันและความหมายของสิ่งที่ฟอนเทนทำเพื่อคนทั่วไปเป็นอย่างดี"
    tokenized_text = tokenizer.tokenize(text)
    assert tokenized_text == ['▁ฉัน', 'เข้าใจ', 'คุณค่า', 'ของ', 'งาน', 'ของ', 'ฉัน', 'และ', 'ความหมาย', 'ของ', 'สิ่ง', 'ที่', 'ฟ', 'อน', 'เท', 'น', 'ทำ', 'เพื่อ', 'คน', 'ทั่วไป', 'เป็นอย่างดี']

def test_thai_text_to_phonemes():
    text = "สวัสดีครับ"
    phonemes = thai_text_to_phonemes(text)
    assert phonemes == "ส ว ั ส ด ี ค ร ั บ"

def test_g2p():
    text = "ฉันรักเมืองไทย"
    normalized_text = text_normalize(text)
    phones, tones, word2ph = g2p(normalized_text)
    assert phones == ['_', 'ฉ', 'ั', 'น', 'ร', 'ั', 'ก', 'เ', 'ม', 'ื', 'อ', 'ง', 'ไ', 'ท', 'ย', '_']
    assert tones == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    assert word2ph == [1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

def test_get_bert_feature():
    text = "ฉันเข้าใจคุณค่าของงานของฉันและความหมายของสิ่งที่ฟอนเทนทำเพื่อคนทั่วไปเป็นอย่างดี"
    normalized_text = text_normalize(text)
    phones, tones, word2ph = g2p(normalized_text)
    bert_features = get_bert_feature(normalized_text, word2ph, device='cpu')
    assert bert_features.shape[0] == len(word2ph)
    assert bert_features.shape[1] == 768  # Assuming hidden size of 768
