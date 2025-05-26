import pytest
import os
import soundfile as sf
import shutil # For cleaning up directories/files

# Assuming export_onnx.py is in the root and contains export_model_to_onnx function
# Adjust path if necessary, e.g., by adding root to sys.path or making export_onnx a callable module
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from export_onnx import export_model_to_onnx 
from melo.api import TTS_ONNX, TTS # TTS is needed to get a valid speaker_id from hps

# Define a temporary directory for test artifacts
TEST_ARTIFACTS_DIR = "test_onnx_artifacts"
DEFAULT_ONNX_MODEL_PATH = os.path.join(TEST_ARTIFACTS_DIR, "test_model.onnx")
DEFAULT_OUTPUT_AUDIO_PATH = os.path.join(TEST_ARTIFACTS_DIR, "test_output.wav")
DEFAULT_LANGUAGE = "EN" # Use a language known to have downloadable models

@pytest.fixture(scope="module")
def onnx_model_path_and_speaker_id():
    # Setup: Create the directory and export ONNX model once per module
    if not os.path.exists(TEST_ARTIFACTS_DIR):
        os.makedirs(TEST_ARTIFACTS_DIR)

    print(f"Exporting ONNX model for language {DEFAULT_LANGUAGE} to {DEFAULT_ONNX_MODEL_PATH}...")
    try:
        # We need hps from a PyTorch model to get a speaker ID
        # This assumes TTS can download the model for DEFAULT_LANGUAGE
        pytorch_model_for_hps = TTS(language=DEFAULT_LANGUAGE, device='cpu', use_hf=True)
        speaker_id = list(pytorch_model_for_hps.hps.data.spk2id.values())[0] # Get the first speaker ID
        del pytorch_model_for_hps # Free up memory

        export_model_to_onnx(
            language=DEFAULT_LANGUAGE, 
            output_path=DEFAULT_ONNX_MODEL_PATH, 
            device='cpu'
            # ckpt_path can be None if using default downloaded models
        )
        print(f"ONNX model exported successfully.")
        yield DEFAULT_ONNX_MODEL_PATH, speaker_id
    except Exception as e:
        print(f"Error during ONNX model export fixture setup: {e}")
        # If export fails, try to provide a dummy path to avoid crashing other tests, 
        # but the test using it will likely fail.
        # Or, re-raise to fail fast:
        raise
    finally:
        # Teardown: Remove the artifacts directory after all tests in the module run
        # print(f"Cleaning up test artifacts directory: {TEST_ARTIFACTS_DIR}")
        # shutil.rmtree(TEST_ARTIFACTS_DIR, ignore_errors=True)
        # Let's not clean up immediately to allow inspection of generated files if tests fail.
        # Cleanup can be manual or handled by a separate script/CI step.
        pass


def test_onnx_tts_generation(onnx_model_path_and_speaker_id):
    onnx_model_path, speaker_id = onnx_model_path_and_speaker_id
    
    if not os.path.exists(onnx_model_path):
        pytest.fail(f"ONNX model file not found at {onnx_model_path}, skipping test. Export might have failed.")

    print(f"Initializing TTS_ONNX with model: {onnx_model_path}")
    onnx_tts_model = TTS_ONNX(
        language=DEFAULT_LANGUAGE, # Must match the language model was exported for
        onnx_path=onnx_model_path,
        device='cpu' 
    )
    
    test_text = "Hello world, this is an ONNX test."
    print(f"Generating audio for text: '{test_text}' using ONNX...")
    
    try:
        onnx_tts_model.tts_to_file(
            text=test_text, 
            speaker_id=speaker_id, # Use the fetched speaker_id
            output_path=DEFAULT_OUTPUT_AUDIO_PATH,
            quiet=True
        )
        print(f"Audio generated at: {DEFAULT_OUTPUT_AUDIO_PATH}")
        
        assert os.path.exists(DEFAULT_OUTPUT_AUDIO_PATH), "Output audio file was not created."
        
        # Check if the audio file has content
        audio_data, sr = sf.read(DEFAULT_OUTPUT_AUDIO_PATH)
        assert len(audio_data) > 0, "Output audio file is empty."
        assert sr == onnx_tts_model.hps.data.sampling_rate, "Output audio sample rate does not match model."
        print("ONNX TTS generation test passed.")

    except Exception as e:
        pytest.fail(f"ONNX TTS generation failed: {e}")
