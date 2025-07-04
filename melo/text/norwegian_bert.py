import torch
from transformers import AutoModel

# Cache for loaded BERT models to avoid reloading from disk/web repeatedly
_model_cache = {}

def get_bert_feature(text: str, word2ph: list[int], tokenizer, model_id: str, device):
    """
    Extracts BERT features aligned with phonemes.

    Args:
        text (str): The input text (normalized).
        word2ph (list[int]): A list where word2ph[i] is the number of phonemes
                             corresponding to the i-th token from tokenizer.tokenize(text).
                             The sum of elements in word2ph is the total number of phonemes.
        tokenizer: The Hugging Face tokenizer instance (e.g., for NbAiLab/nb-bert-base).
        model_id (str): The Hugging Face model ID for the Norwegian BERT model.
        device: The torch device (e.g., 'cpu', 'cuda').

    Returns:
        torch.Tensor: A tensor of shape (total_phonemes, hidden_size)
    """
    global _model_cache

    if model_id not in _model_cache:
        try:
            model = AutoModel.from_pretrained(model_id)
            _model_cache[model_id] = model
        except Exception as e:
            print(f"Error loading BERT model {model_id}: {e}")
            # Fallback or raise error
            # For now, let's re-raise to make it explicit if model loading fails.
            raise e

    model = _model_cache[model_id]
    model.to(device)
    model.eval()

    with torch.no_grad():
        # Tokenize the text, adding special tokens like [CLS] and [SEP]
        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=tokenizer.model_max_length if hasattr(tokenizer, 'model_max_length') else 512
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        try:
            outputs = model(**inputs)
            # last_hidden_state shape: (batch_size, sequence_length, hidden_size)
            bert_hidden_states = outputs.last_hidden_state.squeeze(0)  # Remove batch dimension
        except Exception as e:
            print(f"Error during BERT model forward pass: {e}")
            # Handle error, e.g., by returning None or zeros of expected shape if possible, or re-raise
            # This depends on how the main application can handle missing BERT features.
            # sum_word2ph = sum(word2ph)
            # hidden_size = model.config.hidden_size if hasattr(model, 'config') else 768
            # return torch.zeros((sum_word2ph, hidden_size), device=device)
            raise e


        # The tokenizer adds [CLS] at the beginning and [SEP] at the end.
        # The hidden states corresponding to these special tokens should typically be excluded
        # if word2ph is based on the original tokens without CLS/SEP.
        # word2ph is derived from tokenizer.tokenize(text), which does NOT add special tokens by default.
        # So, the length of word2ph should match the number of actual content tokens.
        # Thus, we slice bert_hidden_states to exclude [CLS] and [SEP].

        # sequence_length from BERT includes [CLS] and [SEP]
        # bert_hidden_states_content = bert_hidden_states[1:-1] # Exclude [CLS] and [SEP]

        # Check if the number of tokens from tokenizer.tokenize() matches the sliced BERT output
        # original_tokens = tokenizer.tokenize(text) # These are the tokens word2ph corresponds to
        # if len(original_tokens) != bert_hidden_states_content.shape[0]:
        #     # This can happen if text is too long and gets truncated by tokenizer for BERT,
        #     # or if padding/special token handling is different than expected.
        #     # Or if tokenizer in g2p and here behave differently regarding max_length.
        #     print(f"Warning: Mismatch in token lengths. len(original_tokens)={len(original_tokens)}, "
        #           f"bert_hidden_states_content.shape[0]={bert_hidden_states_content.shape[0]}. "
        #           f"This might lead to incorrect BERT feature alignment.")
        #     # Attempt to use the shorter length if there's a mismatch due to truncation by BERT tokenizer
        #     # This is a heuristic and might not always be correct.
        #     min_len = min(len(word2ph), bert_hidden_states_content.shape[0])
        #     word2ph_adjusted = word2ph[:min_len]
        #     bert_to_align = bert_hidden_states_content[:min_len]
        # else:
        #     word2ph_adjusted = word2ph
        #     bert_to_align = bert_hidden_states_content

        # More robust handling: align word2ph with the actual input_ids used by BERT, excluding special tokens.
        input_ids_no_special = inputs['input_ids'].squeeze().tolist()
        # Filter out special token IDs (CLS, SEP, PAD)
        # This assumes tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id are available
        content_token_indices = []
        if hasattr(tokenizer, 'cls_token_id') and hasattr(tokenizer, 'sep_token_id') and hasattr(tokenizer, 'pad_token_id'):
            cls_id = tokenizer.cls_token_id
            sep_id = tokenizer.sep_token_id
            pad_id = tokenizer.pad_token_id
            for i, token_id in enumerate(input_ids_no_special):
                if token_id not in [cls_id, sep_id, pad_id]:
                    content_token_indices.append(i)
        else: # Fallback if specific token IDs aren't directly available (less common for modern tokenizers)
            # Default to slicing [CLS] and [SEP] assuming they are at fixed positions and no internal padding tokens matter
            content_token_indices = list(range(1, bert_hidden_states.shape[0] - 1))


        bert_content_features = bert_hidden_states[content_token_indices, :]

        if len(word2ph) != bert_content_features.shape[0]:
             print(f"CRITICAL WARNING: Mismatch after trying to align tokens. "
                   f"len(word2ph)={len(word2ph)}, bert_content_features.shape[0]={bert_content_features.shape[0]}. "
                   f"Text: '{text[:100]}...' "
                   f"This indicates a fundamental issue in token alignment for BERT features.")
             # Fallback strategy: if lengths mismatch significantly, it might be safer to return zeros
             # or attempt a simpler slicing, though this is risky.
             # For now, proceed with the shorter length, but this needs to be fixed if it occurs.
             min_len = min(len(word2ph), bert_content_features.shape[0])
             word2ph_adjusted = word2ph[:min_len]
             bert_to_align = bert_content_features[:min_len]
        else:
            word2ph_adjusted = word2ph
            bert_to_align = bert_content_features


        aligned_bert_features = []
        for i, num_phonemes_for_this_token in enumerate(word2ph_adjusted):
            # bert_to_align[i] is the feature vector for the i-th content token
            # Repeat this feature vector `num_phonemes_for_this_token` times
            if num_phonemes_for_this_token > 0: # Ensure we only extend if there are phonemes
                aligned_bert_features.extend([bert_to_align[i]] * num_phonemes_for_this_token)

        if not aligned_bert_features:
            # This can happen if word2ph_adjusted is empty or all num_phonemes are 0
            # Return a zero tensor of expected shape (0, hidden_size) or handle as error
            hidden_size = model.config.hidden_size if hasattr(model, 'config') else 768
            return torch.empty((0, hidden_size), device=device)

        return torch.stack(aligned_bert_features)


if __name__ == '__main__':
    # Basic test and example usage (requires a Norwegian BERT model and tokenizer)
    # This is a conceptual test, actual execution requires appropriate setup.
    print("Testing Norwegian BERT feature extraction module...")

    # Mocking dependencies for a standalone test run
    class MockTokenizer:
        def __init__(self):
            self.model_max_length = 512
            self.cls_token_id = 101 # Example
            self.sep_token_id = 102 # Example
            self.pad_token_id = 0   # Example

        def tokenize(self, text):
            return text.split() # Simple whitespace tokenizer for mock

        def __call__(self, text, return_tensors, padding, truncation, max_length):
            tokens = self.tokenize(text)
            # Simulate token IDs for [CLS] token1 token2 ... [SEP] [PAD] ...
            # This is a very simplified mock up.
            input_ids = [self.cls_token_id] + [ord(c) for t in tokens for c in t[:5]] + [self.sep_token_id] # Dummy IDs
            input_ids = input_ids[:max_length]
            padded_input_ids = input_ids + [self.pad_token_id] * (max_length - len(input_ids))

            attention_mask = [1] * len(input_ids) + [0] * (max_length - len(input_ids))

            return {
                "input_ids": torch.tensor([padded_input_ids]),
                "attention_mask": torch.tensor([attention_mask])
            }

    mock_tokenizer = MockTokenizer()
    # Test model ID (replace with a real one if testing with actual model)
    # For local testing without downloading, one might need to mock AutoModel.from_pretrained as well.
    test_model_id = "NbAiLab/nb-bert-base" # or a smaller dummy model if available
    # test_model_id = "prajjwal1/bert-tiny" # A small BERT model for quick testing structure

    sample_text = "Dette er en test."
    # word2ph should come from g2p output for `sample_text` tokenized by `mock_tokenizer`
    # mock_tokenizer.tokenize(sample_text) -> ['Dette', 'er', 'en', 'test.']
    # Assume g2p output gives:
    # 'Dette' -> 4 phonemes
    # 'er'    -> 2 phonemes
    # 'en'    -> 2 phonemes
    # 'test.' -> 4 phonemes
    sample_word2ph = [4, 2, 2, 4]
    device = 'cpu' # Or 'cuda' if available

    print(f"Sample text: '{sample_text}'")
    print(f"Sample word2ph: {sample_word2ph}")

    try:
        # To run this test, you need transformers installed and internet access (or cached model)
        # Also, the model_id should be a valid Hugging Face model ID.
        # For a true unit test, AutoModel.from_pretrained would be mocked.
        print(f"Attempting to use model: {test_model_id}")
        # features = get_bert_feature(sample_text, sample_word2ph, mock_tokenizer, test_model_id, device)
        # print(f"Extracted features shape: {features.shape}")
        # expected_total_phonemes = sum(sample_word2ph)
        # self.assertEqual(features.shape[0], expected_total_phonemes)
        # self.assertTrue(features.shape[1] > 0) # Hidden size
        print("Conceptual test: Actual model loading and feature extraction would run here.")
        print("To run full test: uncomment feature extraction, ensure model is available and tokenizer is real.")

    except Exception as e:
        print(f"Error during conceptual test: {e}")
        print("This might be due to model availability or other setup issues for standalone run.")

    # Example of how it's called from norwegian.py:
    # from . import norwegian_bert
    # from .norwegian import tokenizer_no, MODEL_ID_NO # (tokenizer_no and MODEL_ID_NO are from norwegian.py)
    # features = norwegian_bert.get_bert_feature(normalized_text, word2ph_from_g2p, tokenizer_no, MODEL_ID_NO, device)
    print("\nDone.")Tool output for `create_file_with_block`:
