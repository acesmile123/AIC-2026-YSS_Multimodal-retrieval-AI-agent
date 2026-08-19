import torch
import open_clip


class ClipEncoder:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model, _, _ = open_clip.create_model_and_transforms(
            "ViT-B-32",
            pretrained="openai"
        )

        self.tokenizer = open_clip.get_tokenizer("ViT-B-32")
        self.model = self.model.to(self.device)
        self.model.eval()

    def encode_text(self, text):
        tokens = self.tokenizer([text]).to(self.device)

        with torch.no_grad():
            vector = self.model.encode_text(tokens)

        vector = vector.cpu().numpy().astype("float32")

        return vector[0]