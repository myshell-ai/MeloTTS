import torch
import os
import argparse
import melo.api

def export_model_to_onnx(language: str, ckpt_path: str = None, output_path: str = 'melo.onnx', device: str = 'cpu'):
    """
    Exports the TTS model to ONNX format.

    Args:
        language (str): Language of the model (e.g., 'EN').
        ckpt_path (str, optional): Path to the model checkpoint. Defaults to None.
        output_path (str, optional): Path to save the ONNX model. Defaults to 'melo.onnx'.
        device (str, optional): Device to use for export (e.g., 'cpu', 'cuda'). Defaults to 'cpu'.
    """
    print(f"Loading model for language: {language} with device: {device}")
    if ckpt_path:
        print(f"Using checkpoint: {ckpt_path}")
    
    TTS_model = melo.api.TTS(language=language, ckpt_path=ckpt_path, device=device, use_hf=True)
    model = TTS_model.model
    hps = TTS_model.hps

    print("Model loaded successfully.")

    # Prepare dummy input tensors for model.infer()
    x = torch.randint(0, len(hps.symbols), (1, 50), device=device)
    x_lengths = torch.LongTensor([50], device=device)
    sid = torch.LongTensor([0], device=device)
    
    num_tones_for_dummy = hps.num_tones if hps.num_tones is not None and hps.num_tones > 0 else 1
    tone_input = torch.randint(0, num_tones_for_dummy, (1, 50), device=device)
    
    language_input = torch.LongTensor([0], device=device) # Assuming 0 is a valid language ID for the dummy input
    
    # Adjust bert_input and ja_bert_input based on model configuration
    # These dimensions might need to be verified against the specific model's expected input shapes
    bert_feature_dim = hps.data.text_encoder.inter_channels # Example, verify actual attribute
    ja_bert_feature_dim = 768 # Common dimension for Japanese BERT, verify

    bert_input = torch.randn(1, bert_feature_dim, 50, device=device) if hps.data.use_bert else torch.zeros(1, bert_feature_dim, 50, device=device)
    ja_bert_input = torch.randn(1, ja_bert_feature_dim, 50, device=device) if hps.data.use_japanese_bert else torch.zeros(1, ja_bert_feature_dim, 50, device=device)

    # Default values for inference parameters
    noise_scale = 0.667
    length_scale = 1.0
    noise_scale_w = 0.8
    max_len = None 
    sdp_ratio = 0.0 # sdp_ratio is often 0.0 for VITS models if not using specific features

    args_for_onnx_export = (
        x, x_lengths, sid, tone_input, language_input, bert_input, ja_bert_input,
        noise_scale, length_scale, noise_scale_w, max_len, sdp_ratio
    )

    input_names = ['x', 'x_lengths', 'sid', 'tone', 'language', 'bert', 'ja_bert']
    output_names = ['audio_output']

    dynamic_axes = {
        'x': {1: 'sequence'},
        'tone': {1: 'sequence'},
        'bert': {2: 'sequence'}, # Assuming bert features align with sequence length
        'ja_bert': {2: 'sequence'}, # Assuming ja_bert features align with sequence length
        'audio_output': {2: 'audio_length'}
    }

    print(f"Exporting model to ONNX at {output_path}...")
    torch.onnx.export(
        model,
        args_for_onnx_export,
        output_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes
    )
    print(f"Model exported successfully to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Export MeloTTS model to ONNX format.")
    parser.add_argument('--language', type=str, required=True, help="Language of the model (e.g., 'EN', 'ZH', 'ES', 'FR', 'JA', 'KO').")
    parser.add_argument('--ckpt_path', type=str, default=None, help="Path to the model checkpoint file. If None, downloads the default model for the language.")
    parser.add_argument('--output_path', type=str, default='melo.onnx', help="Path to save the exported ONNX model.")
    parser.add_argument('--device', type=str, default='cpu', help="Device to use for export (e.g., 'cpu', 'cuda').")

    args = parser.parse_args()

    export_model_to_onnx(
        language=args.language,
        ckpt_path=args.ckpt_path,
        output_path=args.output_path,
        device=args.device
    )
