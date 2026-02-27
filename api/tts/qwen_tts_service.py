import re
import unicodedata
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel, Qwen3TTSTokenizer
from pathlib import Path
from typing import List, Optional, Tuple, Union

AudioInput = Union[str, Tuple[torch.Tensor, int]]


class QwenTTSService:
    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.model = None
        self.processor = None
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    def slugify_filename(text: str) -> str:
        """
        Создаёт безопасное имя файла для любых языков.
        Оставляем буквы (включая кириллицу), цифры, _ и -
        """
        text = text.lower()
        text = unicodedata.normalize("NFKD", text)
        text = re.sub(r'[^\w\-]+', '_', text)  # заменяем пробелы и спецсимволы на _
        return text

    def load_model(self, model_name: str):
        model_path = f"{self.model_dir}/{model_name}"

        self.model = Qwen3TTSModel.from_pretrained(
            model_path,               # или "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
            device_map=self.device,
            dtype=torch.float32,      # на Mac без CUDA лучше float32
        )

    def generate_custom_voice(
        self,
        text: Union[str, List[str]],
        language: Optional[Union[str, List[str]]] = "Auto",
        speaker: Optional[Union[str, List[str]]] = None,
        instruct: Optional[Union[str, List[str]]] = None,
        output_paths: Optional[List[str]] = None
    ) -> List[str]:
        wavs, sr = self.model.generate_custom_voice(
            text=text,
            language=language,
            speaker=speaker,
            instruct=instruct
        )

        paths = []
        for i, wav in enumerate(wavs):
            path = output_paths[i] if output_paths else f"output_custom_voice_{i}.wav"
            sf.write(path, wav, sr)
            paths.append(path)

        return paths

    def generate_voice_design(
        self,
        text: Union[str, List[str]],
        language: Optional[Union[str, List[str]]] = "Auto",
        instruct: Optional[Union[str, List[str]]] = None,
        output_paths: Optional[List[str]] = None
    ) -> List[str]:
        wavs, sr = self.model.generate_voice_design(
            text=text,
            language=language,
            instruct=instruct
        )
        paths = []
        for i, wav in enumerate(wavs):
            path = output_paths[i] if output_paths else f"output_voice_design_{i}.wav"
            sf.write(path, wav, sr)
            paths.append(path)
        return paths

    def generate_voice_clone(
        self,
        text: Union[str, List[str]],
        language: Optional[Union[str, List[str]]] = "Auto",
        ref_audio: AudioInput = None,
        ref_text: Optional[Union[str, List[str]]] = None,
        voice_clone_prompt: Optional[dict] = None,
        output_paths: Optional[List[str]] = None
    ) -> List[str]:
        wavs, sr = self.model.generate_voice_clone(
            text=text,
            language=language,
            ref_audio=ref_audio,
            ref_text=ref_text,
            voice_clone_prompt=voice_clone_prompt
        )
        paths = []
        for i, wav in enumerate(wavs):
            path = output_paths[i] if output_paths else f"output_voice_clone_{i}.wav"
            sf.write(path, wav, sr)
            paths.append(path)
        return paths


    def get_supported_speakers(self):
        speakers = self.model.get_supported_speakers()
        return speakers

    def get_supported_languages(self):
        languages = self.model.get_supported_languages()
        return languages
