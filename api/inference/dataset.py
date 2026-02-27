# dataset.py
from torch.utils.data import Dataset
from PIL import Image
import base64, io


class MiniDataset(Dataset):
    def __init__(self, processor, data):
        self.processor = processor
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        messages = []

        messages.append({
            "role": "user",
            "content": [{"type": "text", "text": item["prompt"]}]
        })
        messages.append({
            "role": "assistant",
            "content": [{"type": "text", "text": item["response"]}]
        })

        if "image_b64" in item and item["image_b64"]:
            image_bytes = base64.b64decode(item["image_b64"])
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

        prompt_len = len(inputs.input_ids[0]) - len(self.processor.tokenizer(item["response"]).input_ids)
        labels[:prompt_len] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": inputs.attention_mask.squeeze(0),
            "labels": labels
        }
