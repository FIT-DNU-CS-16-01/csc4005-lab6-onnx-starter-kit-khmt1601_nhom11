from pathlib import Path

REQUIRED_PATHS = [
    "README.md",
    "REPORT_TEMPLATE.md",
    "requirements.txt",
    "configs/export_vit_onnx.json",
    "configs/consistency_test.json",
    "configs/benchmark_cpu.json",
    "docs/LAB_GUIDE_LAB6_ONNX.md",
    "docs/ONNX_GUIDE.md",
    "docs/GITHUB_CLASSROOM_GUIDE.md",
    "docs/TROUBLESHOOTING.md",
    "src/__init__.py",
    "src/model.py",
    "src/dataset.py",
    "src/runtime.py",
    "src/export_onnx.py",
    "src/infer_pytorch.py",
    "src/infer_onnx.py",
    "src/consistency_test.py",
    "src/benchmark.py",
    "src/utils.py",
    "ci/smoke_export.py",
]

def main() -> None:
    missing = [p for p in REQUIRED_PATHS if not Path(p).exists()]
    if missing:
        print("Missing required files:")
        for p in missing:
            print(f"- {p}")
        raise SystemExit(1)
    print("Structure check passed.")

if __name__ == "__main__":
    main()
