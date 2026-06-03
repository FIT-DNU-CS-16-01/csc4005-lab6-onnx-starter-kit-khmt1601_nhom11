from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.model import build_model, infer_num_classes_from_checkpoint_metadata, load_checkpoint_state
from src.runtime import create_onnx_session, run_onnx
from src.utils import file_size_mb, percentile, save_json, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark PyTorch and ONNXRuntime inference.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--onnx_path", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default=None, help="Reserved for future real-batch benchmark.")
    parser.add_argument("--model_name", type=str, default="vit_b_16")
    parser.add_argument("--num_classes", type=int, default=5)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_sizes", nargs="+", type=int, default=[1, 4, 8])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def measure_pytorch(model, batch, warmup: int, repeat: int) -> list[float]:
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(batch)

        times = []
        for _ in range(repeat):
            start = time.perf_counter()
            _ = model(batch)
            end = time.perf_counter()
            times.append((end - start) * 1000.0)
    return times


def measure_onnx(session, batch_np, warmup: int, repeat: int) -> list[float]:
    for _ in range(warmup):
        _ = run_onnx(session, batch_np)

    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        _ = run_onnx(session, batch_np)
        end = time.perf_counter()
        times.append((end - start) * 1000.0)
    return times


def summarize(runtime: str, batch_size: int, times_ms: list[float], model_size_mb: float) -> dict:
    mean_latency = float(np.mean(times_ms))
    median_latency = float(np.median(times_ms))
    p95_latency = percentile(times_ms, 95)
    throughput = float(batch_size / (mean_latency / 1000.0))
    return {
        "runtime": runtime,
        "batch_size": batch_size,
        "mean_latency_ms": mean_latency,
        "median_latency_ms": median_latency,
        "p95_latency_ms": p95_latency,
        "throughput_img_per_sec": throughput,
        "model_size_mb": model_size_mb,
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    checkpoint_meta, state_dict = load_checkpoint_state(args.checkpoint)
    num_classes = infer_num_classes_from_checkpoint_metadata(checkpoint_meta, fallback=args.num_classes)

    model = build_model(args.model_name, num_classes=num_classes, pretrained=False)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    session = create_onnx_session(args.onnx_path)

    checkpoint_size_mb = file_size_mb(args.checkpoint)
    onnx_size_mb = file_size_mb(args.onnx_path)

    rows = []
    for batch_size in args.batch_sizes:
        batch = torch.randn(batch_size, 3, args.img_size, args.img_size)
        batch_np = batch.numpy().astype(np.float32)

        pt_times = measure_pytorch(model, batch, args.warmup, args.repeat)
        onnx_times = measure_onnx(session, batch_np, args.warmup, args.repeat)

        rows.append(summarize("PyTorch", batch_size, pt_times, checkpoint_size_mb))
        rows.append(summarize("ONNXRuntime", batch_size, onnx_times, onnx_size_mb))

    output_dir = Path(args.onnx_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "benchmark_results.csv"
    json_path = output_dir / "benchmark_summary.json"

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)

    summary = {
        "checkpoint": args.checkpoint,
        "onnx_path": args.onnx_path,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "batch_sizes": args.batch_sizes,
        "rows": rows,
    }
    save_json(summary, json_path)

    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
