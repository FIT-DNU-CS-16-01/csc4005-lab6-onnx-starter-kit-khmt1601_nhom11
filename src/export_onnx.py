from __future__ import annotations

import argparse
from pathlib import Path

import torch
import onnx

from src.model import build_model, infer_num_classes_from_checkpoint_metadata, load_checkpoint_state
from src.utils import file_size_mb, save_json, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Export PyTorch ViT checkpoint to ONNX.")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--onnx_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="vit_b_16")
    parser.add_argument("--num_classes", type=int, default=5)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--dynamic_batch", action="store_true")
    parser.add_argument("--input_name", type=str, default="input")
    parser.add_argument("--output_name", type=str, default="logits")
    parser.add_argument("--no_checkpoint", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    checkpoint_meta = {}
    state_dict = None
    if not args.no_checkpoint:
        if args.checkpoint is None:
            raise ValueError("Provide --checkpoint or use --no_checkpoint for debug export.")
        checkpoint_meta, state_dict = load_checkpoint_state(args.checkpoint)
        args.num_classes = infer_num_classes_from_checkpoint_metadata(checkpoint_meta, fallback=args.num_classes)

    model = build_model(
        model_name=args.model_name,
        num_classes=args.num_classes,
        dropout=args.dropout,
        pretrained=False,
        train_mode="head_only",
    )

    if state_dict is not None:
        model.load_state_dict(state_dict, strict=True)

    model.eval()

    dummy_input = torch.randn(1, 3, args.img_size, args.img_size)
    onnx_path = Path(args.onnx_path)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    dynamic_axes = None
    if args.dynamic_batch:
        dynamic_axes = {
            args.input_name: {0: "batch_size"},
            args.output_name: {0: "batch_size"},
        }
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=False,
        input_names=[args.input_name],
        output_names=[args.output_name],
        dynamic_axes=dynamic_axes,
        dynamo=True
    )

    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)

    report = {
        "onnx_path": str(onnx_path),
        "checkpoint": args.checkpoint,
        "model_name": args.model_name,
        "num_classes": args.num_classes,
        "img_size": args.img_size,
        "opset": args.opset,
        "dynamic_batch": bool(args.dynamic_batch),
        "input_name": args.input_name,
        "output_name": args.output_name,
        "onnx_size_mb": file_size_mb(onnx_path),
        "status": "exported_and_checked",
    }
    save_json(report, onnx_path.parent / "export_report.json")
    print(report)


if __name__ == "__main__":
    main()
