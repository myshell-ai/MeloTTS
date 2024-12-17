from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import io
from melo.api import TTS
from fastapi.responses import StreamingResponse

app = FastAPI()

# Initialize the TTS models as before
device = 'auto'
models = {
    'EN': TTS(language='EN', device=device),
    'ES': TTS(language='ES', device=device),
    'FR': TTS(language='FR', device=device),
    'ZH': TTS(language='ZH', device=device),
    'JP': TTS(language='JP', device=device),
    'KR': TTS(language='KR', device=device),
}

class SynthesizePayload(BaseModel):
    text: str = 'Ahoy there matey! There she blows!'
    language: str = 'EN'
    speaker: str = 'EN-US'
    speed: float = 1.0

@app.post("/stream")
async def synthesize_stream(payload: SynthesizePayload):
    language = payload.language
    text = payload.text
    speaker = payload.speaker or list(models[language].hps.data.spk2id.keys())[0]
    speed = payload.speed

    def audio_stream():
        bio = io.BytesIO()
        models[language].tts_to_file(text, models[language].hps.data.spk2id[speaker], bio, speed=speed, format='wav')
        audio_data = bio.getvalue()
        yield audio_data

    return StreamingResponse(audio_stream(), media_type="audio/wav")