from __future__ import annotations

import subprocess
import sys


def main() -> None:
    cmd = [
        sys.executable,
        "-m",
        "src.export_onnx",
        "--onnx_path",
        "outputs/debug_random_vit.onnx",
        "--model_name",
        "vit_b_16",
        "--num_classes",
        "5",
        "--img_size",
        "224",
        "--opset",
        "17",
        "--dynamic_batch",
        "--no_checkpoint",
    ]
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
