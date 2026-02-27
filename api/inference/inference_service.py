import torch
import base64
import io
from PIL import Image
from typing import List, Optional, Dict, Any
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from peft import LoraConfig, get_peft_model


class QwenInferenceService:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.processor = None
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            torch.device("cpu")

    def load(self):
        print(f"Loading model on device: {self.device}")
        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            local_files_only=True  # <- local
        )
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16 if self.device.type == "mps" else torch.float32,
            device_map={"": self.device} if self.device.type == "mps" else None
        )
        print("Model loaded successfully")

    def enable_lora(self):
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=[
                "q_proj",
                "v_proj"
            ],
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM"
        )
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

    def train_lora(self, dataset, epochs=1, lr=2e-4, batch_size=1):
        """
        dataset: torch Dataset, который отдаёт input_ids, attention_mask, labels
        """
        self.model.train()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)

        from torch.utils.data import DataLoader
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            total_loss = 0
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids.unsqueeze(0),
                    attention_mask=attention_mask.unsqueeze(0),
                    labels=labels.unsqueeze(0)
                )

                loss = outputs.loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                total_loss += loss.item()

            print(f"[LoRA Train] Epoch {epoch+1}/{epochs},"
                  f"Loss: {total_loss/len(dataloader):.4f}")

    def _decode_image(self, image_b64: str) -> Image.Image:
        image_bytes = base64.b64decode(image_b64)
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")

    def build_messages(
        self,
        prompt: str,
        instructions: Optional[List[str]],
        image_b64: Optional[str]
    ):
        messages = []

        # system messages first
        for instr in instructions or []:
            messages.append({
                "role": "system",
                "content": [{"type": "text", "text": instr}]
            })

        content = [{"type": "text", "text": prompt}]

        if image_b64:
            image = self._decode_image(image_b64)
            content.insert(0, {"type": "image", "image": image})

        messages.append({
            "role": "user",
            "content": content
        })

        return messages

    def generate(
        self,
        prompt: str,
        instructions: Optional[List[str]] = None,
        image_b64: Optional[str] = None,
        gen_config: Optional[Dict[str, Any]] = None
    ):
        gen_config = gen_config or {
            "max_new_tokens": 256,
            "temperature": 0.2,
            "top_p": 0.9
        }

        messages = self.build_messages(prompt, instructions, image_b64)

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                **gen_config
            )

        trimmed = [
            out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)
        ]

        output = self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )

        return output[0]
