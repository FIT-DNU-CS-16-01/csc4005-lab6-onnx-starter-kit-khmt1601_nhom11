[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/1Gv1b95C)
# CSC4005 Lab 6 – Export ONNX + Consistency Test + Benchmark

Starter kit này dành cho **Lab 6 – Export ONNX + Consistency Test + Benchmark** của học phần **CSC4005 – Học sâu**.

> Lab này nối tiếp trực tiếp case study **Smart Campus Scene Classification with Vision Transformer**. Sinh viên sử dụng checkpoint `best_model.pt` đã huấn luyện ở lab ViT trước đó, export mô hình PyTorch sang ONNX, kiểm thử độ nhất quán đầu ra PyTorch–ONNX, sau đó benchmark latency, throughput và model size.

## 1. Mục tiêu

Sau lab này, sinh viên cần:

1. export được mô hình PyTorch sang định dạng `.onnx`;
2. chạy inference bằng `onnxruntime`;
3. kiểm thử consistency giữa PyTorch và ONNX;
4. benchmark latency/throughput/model size;
5. giải thích được vì sao cần `model.eval()`, warm-up khi benchmark, và dynamic batch khi export;
6. viết báo cáo ngắn về kết quả triển khai mô hình.

## 2. Bối cảnh bài toán

Case study:

```text
Smart Campus Scene Classification
```

Mô hình nhận ảnh một không gian trong trường và dự đoán một trong 5 lớp:

```text
classroom
computerroom
library
corridor
office
```

Ở lab trước, mô hình được huấn luyện bằng Vision Transformer. Ở lab này, ta đặt câu hỏi:

> Mô hình PyTorch đã train xong, nhưng nếu muốn đưa vào môi trường triển khai thì cần làm gì?

Luồng thực hành:

```text
best_model.pt
    ↓
export_onnx.py
    ↓
model.onnx
    ↓
consistency_test.py
    ↓
benchmark.py
    ↓
bảng latency / throughput / model size
```

## 3. Cấu trúc repo

```text
csc4005_lab6_onnx_consistency_benchmark_starter/
├── README.md
├── REPORT_TEMPLATE.md
├── requirements.txt
├── configs/
│   ├── export_vit_onnx.json
│   ├── consistency_test.json
│   └── benchmark_cpu.json
├── docs/
│   ├── LAB_GUIDE_LAB6_ONNX.md
│   ├── ONNX_GUIDE.md
│   ├── GITHUB_CLASSROOM_GUIDE.md
│   └── TROUBLESHOOTING.md
├── notebooks/
│   └── lab6_onnx_demo.ipynb
├── src/
│   ├── __init__.py
│   ├── dataset.py
│   ├── model.py
│   ├── runtime.py
│   ├── export_onnx.py
│   ├── infer_pytorch.py
│   ├── infer_onnx.py
│   ├── consistency_test.py
│   ├── benchmark.py
│   └── utils.py
├── ci/
│   ├── check_structure.py
│   └── smoke_export.py
├── checkpoints/
├── outputs/
└── .github/
    └── workflows/
        └── ci.yml
```

## 4. Chuẩn bị môi trường

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 5. Chuẩn bị checkpoint và dữ liệu

Repo này **không chứa dataset và checkpoint**.

Sinh viên cần chuẩn bị:

```text
checkpoints/best_model.pt
data/mit_indoor_smartcampus_5/
```

Ví dụ cấu trúc dữ liệu:

```text
data/mit_indoor_smartcampus_5/
├── classroom/
├── computerroom/
├── library/
├── corridor/
└── office/
```

Checkpoint `best_model.pt` nên lấy từ lab ViT trước đó.

## 6. Bước 1 – Export ONNX

```bash
python -m src.export_onnx \
  --checkpoint checkpoints/best_model.pt \
  --onnx_path outputs/vit_smartcampus.onnx \
  --model_name vit_b_16 \
  --img_size 224 \
  --opset 17 \
  --dynamic_batch
```

Output kỳ vọng:

```text
outputs/vit_smartcampus.onnx
outputs/export_report.json
```

## 7. Bước 2 – Kiểm thử consistency PyTorch vs ONNX

```bash
python -m src.consistency_test \
  --checkpoint checkpoints/best_model.pt \
  --onnx_path outputs/vit_smartcampus.onnx \
  --data_dir data/mit_indoor_smartcampus_5 \
  --num_samples 32 \
  --batch_size 8 \
  --atol 1e-4 \
  --rtol 1e-3
```

Output kỳ vọng:

```text
outputs/consistency_report.json
```

Một kết quả tốt cần có:

```text
passed = true
max_abs_diff nhỏ hơn ngưỡng
mean_abs_diff nhỏ
pred_match_rate gần 1.0
```

## 8. Bước 3 – Benchmark latency / throughput / model size

```bash
python -m src.benchmark \
  --checkpoint checkpoints/best_model.pt \
  --onnx_path outputs/vit_smartcampus.onnx \
  --data_dir data/mit_indoor_smartcampus_5 \
  --batch_sizes 1 4 8 \
  --warmup 10 \
  --repeat 50
```

Output kỳ vọng:

```text
outputs/benchmark_results.csv
outputs/benchmark_summary.json
```

Các metric chính:

```text
runtime
batch_size
mean_latency_ms
median_latency_ms
p95_latency_ms
throughput_img_per_sec
model_size_mb
```

## 9. Chạy thử không cần checkpoint thật

Lệnh này chỉ dùng để kiểm tra pipeline export với mô hình khởi tạo ngẫu nhiên, không dùng để nộp bài chính:

```bash
python -m src.export_onnx \
  --onnx_path outputs/debug_random_vit.onnx \
  --model_name vit_b_16 \
  --num_classes 5 \
  --img_size 224 \
  --opset 17 \
  --dynamic_batch \
  --no_checkpoint
```

## 10. Checklist nộp bài

Sinh viên cần nộp repo có đủ:

```text
outputs/vit_smartcampus.onnx hoặc mô tả nơi lưu nếu file quá lớn
outputs/export_report.json
outputs/consistency_report.json
outputs/benchmark_results.csv
outputs/benchmark_summary.json
REPORT_TEMPLATE.md đã điền hoặc báo cáo riêng
README.md có lệnh chạy tái lập
```

Không nên commit file `.pt`, `.onnx`, dataset hoặc output quá lớn nếu GitHub báo vượt dung lượng. Nếu file lớn, ghi rõ đường dẫn Google Drive/OneDrive trong báo cáo.

## 11. Lỗi thường gặp

| Lỗi | Nguyên nhân | Cách xử lý |
|---|---|---|
| ONNX output lệch PyTorch | Quên `model.eval()` hoặc preprocessing khác nhau | Đảm bảo dùng cùng transform và `eval()` |
| Benchmark số liệu rất dao động | Không warm-up hoặc repeat quá ít | Dùng `--warmup 10 --repeat 50` |
| Export lỗi dynamic shape | Khai báo dynamic axes sai | Dùng `--dynamic_batch` mặc định |
| ONNXRuntime không import được | Chưa cài `onnxruntime` | Chạy lại `pip install -r requirements.txt` |
| File ONNX quá lớn | ViT-base có nhiều tham số | Không commit file lớn; chỉ nộp report và link lưu trữ |
