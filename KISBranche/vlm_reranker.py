import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor


MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"


class VLMReranker:
    def __init__(self):

        print("[VLM] Loading model...")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print("[VLM] Device:", self.device)

        self.processor = AutoProcessor.from_pretrained(
            MODEL_NAME
        )

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        self.model.eval()

        print("[VLM] Model loaded")


    def score(self, image_path, query, caption=None):

        image = Image.open(image_path).convert("RGB")

        if caption is None:
            caption = ""

        prompt = f"""
You are a video retrieval reranker.

Query:
{query}

Caption:
{caption}

Look carefully at the image.

Determine how relevant this image is to the query.

Return ONLY a number from 0 to 10.

0 = completely irrelevant
5 = somewhat relevant
10 = highly relevant
"""

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt"
        )

        inputs = {
            k: v.to(self.model.device)
            for k, v in inputs.items()
        }

        with torch.no_grad():

            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=10
            )
        generated_ids = [
            output_ids[i][inputs["input_ids"].shape[1]:]
            for i in range(len(output_ids))
        ]
        response = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0]

        print("[VLM RAW]:", response)

        try:
            score = float(response.strip())
        except:
            score = 0.0
        score = max(0.0, min(10.0, score))
        return score