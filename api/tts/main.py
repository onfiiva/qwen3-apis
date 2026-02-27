import io
import os
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
import re
from typing import Optional, List, Union
import soundfile as sf
from urllib.parse import quote

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .schemas import TTSRequest, VoiceCloneRequest
from .qwen_tts_service import QwenTTSService, AudioInput

# Путь к модели
MODEL_DIR = os.path.join(os.path.dirname(__file__), "../../models/tts")
MODEL_NAME = os.environ.get("TTS_MODEL_NAME", "Qwen3-TTS-12Hz-0.6B-CustomVoice")

# Инициализация сервиса
tts_service = QwenTTSService(MODEL_DIR)

# =======================
# Lifespan: загрузка модели
# =======================
@asynccontextmanager
async def lifespan(app: FastAPI):
    tts_service.load_model(MODEL_NAME)
    yield

# =======================
# FastAPI приложение
# =======================
app = FastAPI(
    title="Qwen TTS API",
    lifespan=lifespan
)


# =======================
# Эндпоинты
# =======================
@app.post("/tts/custom_voice")
async def custom_voice(req: TTSRequest):
    try:
        # Генерация аудио в временный файл
        safe_text = QwenTTSService.slugify_filename(req.text)
        output_path = f"/tmp/{safe_text}_{req.speaker}.wav"

        # Сервис сохранит WAV на диск
        wav_path = await asyncio.to_thread(
            tts_service.generate_custom_voice,
            text=req.text,
            language=req.language,
            speaker=req.speaker,
            instruct=req.instruct,
            output_paths=[output_path]
        )

        # Читаем файл
        with open(output_path, "rb") as f:
            audio_bytes = f.read()

        filename = f"{safe_text}_{req.speaker}.wav"
        quoted_filename = quote(filename)

        # Отдаём через StreamingResponse
        return StreamingResponse(
            io.BytesIO(audio_bytes),
            media_type="audio/wav",
            headers={"Content-Disposition": f"attachment; filename*=UTF-8''{quoted_filename}"}
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/tts/voice_design")
async def voice_design(req: TTSRequest):
    try:
        wav_paths = await asyncio.to_thread(
            tts_service.generate_voice_design,
            text=req.text,
            language=req.language,
            instruct=req.instruct,
            output_paths=req.output_paths
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"audio_paths": wav_paths, "provider": "qwen-tts"}

@app.post("/tts/voice_clone")
async def voice_clone(
    text: str,
    ref_audio: Optional[UploadFile] = File(None),
    ref_text: Optional[str] = None,
    language: Optional[str] = "Auto",
    instruct: Optional[str] = None
):
    # Сохраняем временный файл, если прислали UploadFile
    ref_audio_path = None
    if ref_audio:
        ref_audio_path = f"/tmp/{ref_audio.filename}"
        with open(ref_audio_path, "wb") as f:
            f.write(await ref_audio.read())

    try:
        wav_paths = await asyncio.to_thread(
            tts_service.generate_voice_clone,
            text=text,
            language=language,
            ref_audio=ref_audio_path,
            ref_text=ref_text
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"audio_paths": wav_paths, "provider": "qwen-tts"}


@app.get("/get_speakers")
async def get_supported_speakers():
    try:
        speakers = await asyncio.to_thread(
            tts_service.get_supported_speakers
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"speakers": speakers, "provider": "qwen-tts"}


@app.get("/get_languages")
async def get_supported_languages():
    try:
        languages = await asyncio.to_thread(
            tts_service.get_supported_languages
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"languages": languages, "provider": "qwen-tts"}


@app.get("/health")
def health():
    return {"status": "ok"}
