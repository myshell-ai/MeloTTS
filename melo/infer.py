import os
import click
from melo.api import TTS, TTS_ONNX


@click.command()
@click.option('--text', '-t', type=str, default=None, help="Text to speak")
@click.option('--language', '-l', type=str, default="EN", help="Language of the model")
@click.option('--output_dir', '-o', type=str, default="outputs", help="Path to the output directory")
@click.option('--use_onnx/--no-use_onnx', default=False, help="Use ONNX model for inference.")
@click.option('--onnx_path', type=str, default='melo.onnx', help="Path to the ONNX model file (required if --use_onnx).")
@click.option('--ckpt_path', '-m', type=str, default=None, help="Path to the PyTorch checkpoint file (required if not using ONNX).")
@click.option('--config_path', type=str, default=None, help="Path to the model config.json file. Optional, will be inferred or downloaded if not set.")
def main(text, language, output_dir, use_onnx, onnx_path, ckpt_path, config_path):
    if text is None:
        raise ValueError("The --text option must be specified.")

    if use_onnx:
        if onnx_path is None: # Should not happen if default is set, but good practice
            raise ValueError("The --onnx_path must be specified when using --use_onnx.")
        model = TTS_ONNX(
            language=language, 
            onnx_path=onnx_path, 
            config_path=config_path, 
            device='auto', 
            use_hf=True
        )
        print(f"Using ONNX model: {onnx_path}")
    else:
        if ckpt_path is None:
            raise ValueError("The --ckpt_path must be specified if not using ONNX.")
        # If config_path is not provided, TTS will try to infer it from ckpt_path or download
        model = TTS(
            language=language, 
            ckpt_path=ckpt_path, 
            config_path=config_path, 
            device='auto', 
            use_hf=True
        )
        print(f"Using PyTorch model: {ckpt_path}")
    
    # Ensure model was loaded
    if model is None or model.hps is None:
        raise RuntimeError("Model could not be loaded. Check paths and configurations.")

    # Ensure spk2id is available
    if not hasattr(model.hps.data, 'spk2id') or not model.hps.data.spk2id:
        # Fallback for models that might not have spk2id (e.g., single-speaker models or different config structure)
        # Defaulting to speaker ID 0 for such cases.
        # This part might need adjustment based on how speaker IDs are handled in various model configs.
        # For now, we assume a default speaker if spk2id is missing.
        print("Warning: spk2id not found in model config. Using default speaker ID 0.")
        spk_items = {"default_speaker": 0}
    else:
        spk_items = model.hps.data.spk2id.items()

    for spk_name, spk_id in spk_items:
        save_path = f'{output_dir}/{language.lower()}_{spk_name}/output.wav'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        print(f"Synthesizing for speaker: {spk_name} (ID: {spk_id})")
        model.tts_to_file(text, spk_id, save_path, quiet=True) # Added quiet=True to reduce console noise per synthesis

if __name__ == "__main__":
    main()
