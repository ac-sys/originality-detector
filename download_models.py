"""Cross-platform model downloader for Originality Detector (v3 active + v2/v1 rollback)."""
import os

os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from huggingface_hub import snapshot_download

MODELS = [
    # v3 / v2 active
    "fakespot-ai/roberta-base-ai-text-detection-v1",
    "desklib/ai-text-detector-v1.01",
    "intfloat/e5-small-v2",
    # v1 (rollback)
    "distilgpt2",
    "Hello-SimpleAI/chatgpt-detector-roberta",
    "sentence-transformers/all-MiniLM-L6-v2",
]

IGNORE = ["*.msgpack", "*.h5", "*.ot", "*.tflite", "onnx/*", "openvino/*", "coreml/*"]

if __name__ == "__main__":
    for m in MODELS:
        print(f"Downloading {m}...", flush=True)
        snapshot_download(m, ignore_patterns=IGNORE)
        print(f"Finished {m}", flush=True)

