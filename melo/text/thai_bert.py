import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
import sys

models = {}
tokenizers = {}

def get_bert_feature(text, word2ph, device=None, model_id='clicknext/phayathaibert'):
    global model
    global tokenizer

    if (
        sys.platform == "darwin"
        and torch.backends.mps.is_available()
        and device == "cpu"
    ):
        device = "mps"
    if not device:
        device = "cuda"
    if model_id not in models:
        model = AutoModelForMaskedLM.from_pretrained(model_id).to(
            device
        )
        models[model_id] = model
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizers[model_id] = tokenizer
    else:
        model = models[model_id]
        tokenizer = tokenizers[model_id]


    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt")
        tokenized = tokenizer.tokenize(text)
        for i in inputs:
            inputs[i] = inputs[i].to(device)
        res = model(**inputs, output_hidden_states=True)
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()

    try:
        assert inputs["input_ids"].shape[-1] == len(word2ph), f"Input IDs length: {inputs['input_ids'].shape[-1]} / Word2ph length: {len(word2ph)}, Tokenized: {tokenized}, Inputs: {inputs}, Word2ph: {word2ph}"
    except AssertionError as e:
        print(f"Warning: Mismatch in input lengths. Details: {e}")

    word2phone = word2ph
    phone_level_feature = []
    for i in range(len(word2phone)):
        repeat_feature = res[i].repeat(word2phone[i], 1)
        phone_level_feature.append(repeat_feature)

    phone_level_feature = torch.cat(phone_level_feature, dim=0)

    return phone_level_feature.T
