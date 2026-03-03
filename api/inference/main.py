import os
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

from .inference_service import QwenInferenceService
from .schemas import GenerateRequest, TrainRequest
from .dataset import MiniDataset

# prod
# MODEL_PATH = os.environ.get("MODEL_PATH", "/models/inference/qwen3v1")
# local
MODEL_PATH = os.path.join(os.path.dirname(__file__), "../../models/inference/qwen3-vl-4b-instruct")

service = QwenInferenceService(MODEL_PATH)


@asynccontextmanager
async def lifespan(app: FastAPI):
    service.load()
    yield


app = FastAPI(
    title="Qwen Inference API",
    lifespan=lifespan
)


@app.post("/generate")
async def generate(req: GenerateRequest):

    try:
        text = await asyncio.to_thread(
            service.generate,
            req.prompt,
            req.instruction,
            req.image,
            req.gen_config,
            req.mode
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "text": text,
        "finish_reason": "stop",
        "provider": "qwen3vl"
    }


@app.post("/train_lora")
async def train_lora(req: TrainRequest, background_tasks: BackgroundTasks):
    def train_task():
        dataset = MiniDataset(service.processor, req.data)

        service.enable_lora()
        service.train_lora(
            dataset,
            epochs=req.epochs,
            lr=req.lr,
            batch_size=req.batch_size
        )

        service.lora_model.save_pretrained("lora_weights")

    background_tasks.add_task(train_task)
    return {"status": "training started"}


@app.get("/health")
def health():
    return {"status": "ok"}
