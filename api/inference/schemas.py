from pydantic import BaseModel
from typing import List, Dict, Any


class TrainRequest(BaseModel):
    data: List[Dict[str, Any]]  # list of examples {"prompt": ..., "response": ..., "image_b64": ...}
    epochs: int = 1
    lr: float = 2e-4
    batch_size: int = 1