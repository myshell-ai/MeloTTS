#   melotts_api.py                                                                                                                
#
# made with Google Gemini
from fastapi import FastAPI, Response, HTTPException
from pydantic import BaseModel
from melo.api import TTS
import os
import io
import tempfile # <--- ADD THIS IMPORT

# Initialize FastAPI app
app = FastAPI(
    title="MeloTTS API",
    description="A simple API to expose MeloTTS for text-to-speech generation.",
    version="1.0.0",
)

# Load MeloTTS model (you might want to specify your language/model)
print("Loading MeloTTS model...")
# Use 'cuda' if you have an NVIDIA GPU, otherwise 'cpu'
#model = TTS(language='EN', device='cuda') 
model = TTS(language='EN', device='cpu') 
print("MeloTTS model loaded.")

# Define request body for speech synthesis
class SpeechRequest(BaseModel):
    input: str
    voice: str = "EN-BR"
    model: str = "melo-tts-english-us" # Make default explicit
    speed: float = 0.9

# Define response for models/voices (mimicking OpenAI's structure)
class ModelData(BaseModel):
    id: str
    object: str = "model"
    created: int = 1677641200
    owned_by: str = "community"

class ModelsResponse(BaseModel):
    data: list[ModelData]
    object: str = "list"

class VoicesResponse(BaseModel):
    voices: list[str]

@app.get("/v1/models", response_model=ModelsResponse)
async def get_models():
    """
    Returns a list of available TTS models.
    Mimics OpenAI's /v1/models endpoint for compatibility.
    """
    return ModelsResponse(data=[
        ModelData(id="EN-US", object="model"),
        ModelData(id="EN-BR", object="model"),
        ModelData(id="EN_INDIA", object="model"),
        ModelData(id="EN-AU", object="model"),
        ModelData(id="EN-Default", object="model"),
        # You might add other language models here if you load them
    ])

@app.get("/v1/audio/voices", response_model=VoicesResponse)
async def get_voices():
    """
    Returns a list of available voices.
    Mimics a common pattern for TTS voice endpoints.
    """
    # IMPORTANT: This list MUST match the keys in model.hps.data.spk2id
    return VoicesResponse(voices=['EN-US', 'EN-BR', 'EN_INDIA', 'EN-AU', 'EN-Default'])

@app.post("/v1/audio/speech")
async def create_speech(request: SpeechRequest):
    """
    Synthesizes speech from text using MeloTTS.
    Mimics OpenAI's /v1/audio/speech endpoint.
    """
    if not request.input:
        raise HTTPException(status_code=400, detail="Input text is required.")

    if not request.voice:
        raise HTTPException(status_code=400, detail="Voice is required.")

    try:
        # Determine the speaker ID based on the requested voice
        speaker_ids = model.hps.data.spk2id
        if request.voice in speaker_ids:
            speaker_id = speaker_ids[request.voice]
        else:
            # Log a warning if voice not found and fall back to a default
            print(f"Warning: Voice '{request.voice}' not found, falling back to 'EN-BR'")
            speaker_id = speaker_ids.get("EN-BR", 0) # Default to 0 if 'EN-US' not found

        # Use tempfile.NamedTemporaryFile for robust temporary file creation
        # 'delete=True' ensures the file is removed automatically when the 'with' block exits
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=True) as temp_file:
            temp_audio_file_path = temp_file.name

            # Generate audio and save to the temporary file
            model.tts_to_file(
                text=request.input,
                speaker_id=speaker_id,
                output_path=temp_audio_file_path,
                speed=request.speed
            )

            # Seek to the beginning of the temporary file before reading its content
            temp_file.seek(0)
            audio_bytes = temp_file.read()

        # Return audio as MP3
        return Response(content=audio_bytes, media_type="audio/mpeg")

    except Exception as e:
        # IMPORTANT: Print the full traceback for debugging!
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Speech synthesis failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    # Run the FastAPI app
    uvicorn.run(app, host="0.0.0.0", port=8000)
