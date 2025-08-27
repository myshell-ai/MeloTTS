from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tempfile
import os
import uuid
from melo.api import TTS
import uvicorn
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.makedirs("tmp", exist_ok=True)
app = FastAPI(title = "MeloTTS microservice", version = "1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
models = {}
logger.info("Loading models")
languages = ['EN', 'EN_NEWEST', 'EN_V2']
for language in languages:
    try:
        models[language] = TTS(language=language, device = "auto")
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.info(f"print failed to load the models: {e}")

class TTSRequest(BaseModel):
    text: str
    language: str = "EN"
    speaker: str = "EN-US"
    speed: float = 1.1

class TTSResponse(BaseModel):
    audio_id: str
    message: str

temp_files={}

@app.get("/")
async def root():
    return {"message": "Welcome to the MeloTTS microservice", "version": "1.0.0"}


@app.post("/generate", response_model = TTSResponse)
async def generate_speech(request: TTSRequest):
    try:
        if request.language not in models:
            raise HTTPException(
                status_code = 400,
                detail = f"Language {request.language} not supported. Available: {list(models.keys())}"
            )
        model = models[request.language]
        available_speakers = list(model.hps.data.spk2id.keys())
        if request.speaker not in available_speakers:
            raise HTTPException(
                status_code=400,
                detail=f"Speaker {request.speaker} not found. Available: {available_speakers}"
            )
        audio_id = str(uuid.uuid4())
        temp_path = f"tmp/tts_{audio_id}.wav"
        model.tts_to_file(
            text = request.text,
            speaker_id = model.hps.data.spk2id[request.speaker],
            output_path = temp_path,
            speed = request.speed
        )
        temp_files[audio_id] = temp_path

        return TTSResponse(
            audio_id = audio_id,
            message = f"Speech generated successfully for text: '{request.text[:50]}...'"
            )
    except Exception as e:
        raise HTTPException(
            status_code = 500,
            detail = f"Generation failed: {e}"
        )
    
@app.get("/audio/{audio_id}")
async def get_audio(audio_id: str):
    if audio_id not in temp_files:
        raise HTTPException(
            status_code = 404,
            detail = "Audio file not found."
        )
    file_path = temp_files[audio_id]
    if not os.path.exists(file_path):
        raise HTTPException(
            status_code = 404,
            detail = "Audio file no longer exists."
        )
    
    return FileResponse(
        file_path,
        media_type = "audio/wav",
        filename = f"speech_{audio_id}.wav"
    )

@app.delete("/audio/{audio_id}")
async def cleanup_audio(audio_id: str):

    if audio_id in temp_files:
        try:
            os.remove(temp_files[audio_id])
            del temp_files[audio_id]
            return {"message": "Audio file cleaned up"}
        except:
            logger.error(f"Failed to delete audio file {audio_id}: {e}")
    raise HTTPException(status_code=404, detail="Audio file not found")

@app.get("/health")
async def health_check(): 

    return {
        "status": "healthy",
        "loaded_languages": list(models.keys()),
        "version": "1.0.0"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8888)