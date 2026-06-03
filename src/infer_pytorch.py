from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from src.model import build_model, infer_num_classes_from_checkpoint_metadata, load_checkpoint_state


def parse_args():
    parser = argparse.ArgumentParser(description="Run PyTorch inference on one image.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="vit_b_16")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--class_to_idx", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_meta, state_dict = load_checkpoint_state(args.checkpoint)
    num_classes = infer_num_classes_from_checkpoint_metadata(checkpoint_meta, fallback=5)

    model = build_model(args.model_name, num_classes=num_classes, pretrained=False)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    transform = transforms.Compose(
        [
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    image = Image.open(args.image_path).convert("RGB")
    x = transform(image).unsqueeze(0)

    with torch.no_grad():
        logits = model(x)
        pred = int(logits.argmax(dim=1).item())

    idx_to_class = None
    if args.class_to_idx:
        class_to_idx = json.loads(Path(args.class_to_idx).read_text(encoding="utf-8"))
        idx_to_class = {v: k for k, v in class_to_idx.items()}

    print({"pred_idx": pred, "pred_class": idx_to_class.get(pred) if idx_to_class else None})


if __name__ == "__main__":
    main()
