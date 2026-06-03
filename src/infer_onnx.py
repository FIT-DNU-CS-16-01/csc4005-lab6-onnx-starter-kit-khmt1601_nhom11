from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from torchvision import transforms

from src.runtime import create_onnx_session, run_onnx


def parse_args():
    parser = argparse.ArgumentParser(description="Run ONNXRuntime inference on one image.")
    parser.add_argument("--onnx_path", type=str, required=True)
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--class_to_idx", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    transform = transforms.Compose(
        [
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    image = Image.open(args.image_path).convert("RGB")
    x = transform(image).unsqueeze(0).numpy().astype(np.float32)

    session = create_onnx_session(args.onnx_path)
    logits = run_onnx(session, x)
    pred = int(np.argmax(logits, axis=1)[0])

    idx_to_class = None
    if args.class_to_idx:
        class_to_idx = json.loads(Path(args.class_to_idx).read_text(encoding="utf-8"))
        idx_to_class = {v: k for k, v in class_to_idx.items()}

    print({"pred_idx": pred, "pred_class": idx_to_class.get(pred) if idx_to_class else None})


if __name__ == "__main__":
    main()
