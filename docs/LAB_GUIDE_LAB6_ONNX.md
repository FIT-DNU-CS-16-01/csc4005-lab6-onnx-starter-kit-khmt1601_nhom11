# Lab Guide – Export ONNX + Consistency Test + Benchmark

## 1. Mục tiêu thực hành

Lab này giúp sinh viên chuyển từ tư duy “train model” sang tư duy “deploy model”.

Sinh viên sẽ thực hiện 3 bước:

```text
Export → Consistency Test → Benchmark
```

## 2. Bước 1: Export

Export là bước chuyển mô hình từ PyTorch sang ONNX:

```text
best_model.pt → model.onnx
```

ONNX giúp mô hình có thể chạy ở nhiều runtime khác nhau, ví dụ ONNXRuntime.

## 3. Bước 2: Consistency Test

Mục tiêu là kiểm tra:

```text
cùng input → PyTorch logits ≈ ONNX logits
```

Không nên chỉ kiểm tra mô hình export ra file thành công. Mô hình export được nhưng output sai vẫn là lỗi nghiêm trọng.

## 4. Bước 3: Benchmark

Benchmark cần có:

```text
warm-up
repeat nhiều lần
batch size rõ ràng
cùng input size
cùng thiết bị đo
```

Không nên đo một lần rồi kết luận.

## 5. Artefact cần nộp

```text
export_report.json
consistency_report.json
benchmark_results.csv
benchmark_summary.json
báo cáo phân tích
```
