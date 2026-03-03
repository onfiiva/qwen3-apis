from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class GenerateRequest(BaseModel):
    prompt: str
    instruction: Optional[List[str]] = Field(default_factory=list)
    gen_config: Optional[Dict[str, Any]] = Field(default_factory=dict)
    image: Optional[str] = None
    mode: str = "base"


class TrainSample(BaseModel):
    instruction: str
    input: Optional[str] = None
    output: str


class TrainRequest(BaseModel):
    data: List[TrainSample]
    epochs: int = 1
    lr: float = 2e-4
    batch_size: int = 1
