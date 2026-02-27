from pydantic import BaseModel, Field
from typing import Optional, List, Union
from fastapi import FastAPI, HTTPException, UploadFile, File


class TTSRequest(BaseModel):
    text: Union[str, List[str]]
    language: Optional[Union[str, List[str]]] = "Auto"
    speaker: Optional[Union[str, List[str]]] = None
    instruct: Optional[Union[str, List[str]]] = None

    model_config = {
        "arbitrary_types_allowed": True
    }

class VoiceCloneRequest(TTSRequest):
    ref_audio: Optional[UploadFile] = None  # теперь через UploadFile
    ref_text: Optional[Union[str, List[str]]] = None
    voice_clone_prompt: Optional[dict] = None

    model_config = {
        "arbitrary_types_allowed": True
    }