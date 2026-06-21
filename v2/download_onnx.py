import os
import urllib.request
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

def download_file(url,dest_path):
    print(f"Downloading {url}...")
    urllib.request.urlretrieve(url,dest_path)
    print(f"Saved to {dest_path}")

class ONNXEmbedder:

    def __init__(self):
        model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")
        model_path = os.path.join(model_dir, "model.onnx")
        tokenizer_path = os.path.join(model_dir, "tokenizer.json")
        if (not os.path.exists(model_path) or not os.path.exists(tokenizer_path)):
            raise FileNotFoundError("ONNX Model files not found. Run download_onnx.py first.")
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.tokenizer.enable_truncation(max_length=256)

    def encode(self,text):
        if(isinstance(text,str)):
            return self.encode_single(text)
        return np.array([self.encode_single(t) for t in text])

    def encode_single(self,text):
        encoded = self.tokenizer.encode(text)
        input_ids = np.array([encoded.ids], dtype=np.int64)
        attention_mask = np.array([encoded.attention_mask], dtype=np.int64)
        token_type_ids = np.array([encoded.type_ids], dtype=np.int64)
        inputs = {"input_ids": input_ids, "attention_mask": attention_mask, "token_type_ids": token_type_ids}
        outputs = self.session.run(None, inputs)

        token_embed = outputs[0]
        mask_expanded = np.expand_dims(attention_mask, -1)
        mask_expanded = np.broadcast_to(mask_expanded, token_embed.shape)

        sum_embed = np.sum(token_embed * mask_expanded, axis=1)
        sum_mask = np.clip(mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
        mean_pooled = sum_embed / sum_mask
        vec = mean_pooled[0]
        norm = np.linalg.norm(vec)
        if (norm > 0):
            vec = vec / norm
        return vec


if __name__=="__main__":
    model_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)),"model")
    os.makedirs(model_dir,exist_ok=True)
    onnx_url="https://huggingface.co/Xenova/all-MiniLM-L6-v2/resolve/main/onnx/model.onnx"
    tokenizer_url="https://huggingface.co/Xenova/all-MiniLM-L6-v2/resolve/main/tokenizer.json"
    download_file(onnx_url,os.path.join(model_dir,"model.onnx"))
    download_file(tokenizer_url,os.path.join(model_dir,"tokenizer.json"))
    print("Download complete")