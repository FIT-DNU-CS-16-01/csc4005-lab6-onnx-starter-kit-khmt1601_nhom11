from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.dataset import ImageFolderSubset
from src.model import build_model, infer_num_classes_from_checkpoint_metadata, load_checkpoint_state
from src.runtime import create_onnx_session, run_onnx
from src.utils import save_json, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Compare PyTorch and ONNXRuntime outputs.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--onnx_path", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--classes", nargs="+", default=["classroom", "computerroom", "library", "corridor", "office"])
    parser.add_argument("--num_samples", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--model_name", type=str, default="vit_b_16")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    checkpoint_meta, state_dict = load_checkpoint_state(args.checkpoint)
    num_classes = infer_num_classes_from_checkpoint_metadata(checkpoint_meta, fallback=len(args.classes))
    model = build_model(args.model_name, num_classes=num_classes, pretrained=False)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    if args.data_dir:
        dataset = ImageFolderSubset(
            data_dir=args.data_dir,
            classes=args.classes,
            img_size=args.img_size,
            max_samples=args.num_samples,
            seed=args.seed,
        )
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
        batches = [x for x, _ in loader]
    else:
        batches = [torch.randn(args.batch_size, 3, args.img_size, args.img_size)]

    session = create_onnx_session(args.onnx_path)

    max_abs_diffs = []
    mean_abs_diffs = []
    total = 0
    pred_match = 0

    with torch.no_grad():
        for batch in batches:
            torch_logits = model(batch).cpu().numpy()
            onnx_logits = run_onnx(session, batch.cpu().numpy())

            diff = np.abs(torch_logits - onnx_logits)
            max_abs_diffs.append(float(diff.max()))
            mean_abs_diffs.append(float(diff.mean()))

            torch_pred = np.argmax(torch_logits, axis=1)
            onnx_pred = np.argmax(onnx_logits, axis=1)
            pred_match += int((torch_pred == onnx_pred).sum())
            total += len(torch_pred)

    max_abs_diff = float(np.max(max_abs_diffs))
    mean_abs_diff = float(np.mean(mean_abs_diffs))
    pred_match_rate = float(pred_match / max(1, total))
    passed = bool(max_abs_diff <= args.atol or np.allclose(0.0, max_abs_diff, atol=args.atol, rtol=args.rtol))

    report = {
        "checkpoint": args.checkpoint,
        "onnx_path": args.onnx_path,
        "num_samples": total,
        "batch_size": args.batch_size,
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
        "pred_match_rate": pred_match_rate,
        "atol": args.atol,
        "rtol": args.rtol,
        "passed": passed,
    }

    output_path = Path(args.onnx_path).parent / "consistency_report.json"
    save_json(report, output_path)
    print(report)


if __name__ == "__main__":
    main()
