# dataset.py
from torch.utils.data import Dataset
from PIL import Image
import base64, io

from api.inference.schemas import TrainSample


class MiniDataset(Dataset):
    def __init__(self, processor, data):
        self.processor = processor
        self.data = data  # list[TrainSample]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item: TrainSample = self.data[idx]  # тип явно Pydantic
        messages = []

        # используем атрибуты Pydantic напрямую
        prompt = item.input or item.instruction  # если input пустой, берем instruction
        response = item.output
        image_b64 = getattr(item, "image_b64", None)  # если решишь добавить image

        messages.append({
            "role": "user",
            "content": [{"type": "text", "text": prompt}]
        })
        messages.append({
            "role": "assistant",
            "content": [{"type": "text", "text": response}]
        })

        if image_b64:
            image_bytes = base64.b64decode(image_b64)
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            messages[0]["content"].insert(0, {"type": "image", "image": image})

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )

        input_ids = inputs.input_ids.squeeze(0)
        labels = input_ids.clone()

        # вычисляем длину prompt, чтобы маскировать в labels
        prompt_len = len(inputs.input_ids[0]) - len(self.processor.tokenizer(response).input_ids)
        labels[:prompt_len] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": inputs.attention_mask.squeeze(0),
            "labels": labels
        }
