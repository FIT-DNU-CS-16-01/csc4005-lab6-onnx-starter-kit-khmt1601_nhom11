# Troubleshooting

## 1. `ModuleNotFoundError: onnxruntime`

Chạy lại:

```bash
pip install -r requirements.txt
```

## 2. Export ONNX bị lỗi do shape

Thử thêm:

```bash
--dynamic_batch
```

và kiểm tra `--img_size` có đúng với lúc train không.

## 3. Output PyTorch và ONNX lệch nhiều

Kiểm tra:

- đã gọi `model.eval()` chưa;
- checkpoint có đúng model không;
- số lớp có đúng không;
- preprocessing có giống nhau không;
- input image size có đúng không.

## 4. Benchmark lúc nhanh lúc chậm

Nên tăng:

```bash
--warmup 10
--repeat 50
```

Đóng bớt ứng dụng khác khi benchmark trên laptop.

## 5. File ONNX quá lớn

Không commit file lớn lên GitHub. Ghi link Google Drive/OneDrive trong báo cáo.
