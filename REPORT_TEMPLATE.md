# CSC4005 Lab 6 Report – Export ONNX + Consistency Test + Benchmark

## 1. Thông tin

- Họ tên: Trần Trường Giang
- Mã sinh viên: 1671040009
- Lớp: KHMT 16-01
- Link GitHub repo:
- Link checkpoint hoặc mô tả checkpoint sử dụng:
- Link file ONNX nếu không commit trực tiếp:

## 2. Mô tả mô hình đầu vào

| Nội dung | Giá trị |
|---|---|
| Bài toán | Smart Campus Scene Classification |
| Dataset | MIT Indoor Scenes 67 subset |
| Số lớp | 5 |
| Classes | classroom, computerroom, library, corridor, office |
| Model PyTorch | ... |
| Checkpoint | ... |
| Image size | ... |
| Train mode từ lab trước | head_only / finetune |

## 3. Export ONNX

Điền thông tin:

| Thông số | Giá trị |
|---|---|
| ONNX path | ... |
| Opset | ... |
| Dynamic batch | yes / no |
| Input name | input |
| Output name | logits |
| Model size | ... MB |

Lệnh đã chạy:

```bash
...
```

## 4. Consistency Test

| Metric | Giá trị |
|---|---:|
| passed | ... |
| num_samples | ... |
| batch_size | ... |
| max_abs_diff | ... |
| mean_abs_diff | ... |
| pred_match_rate | ... |
| atol | ... |
| rtol | ... |

Nhận xét:

- PyTorch và ONNX có nhất quán không?
- Nếu có sai khác, sai khác lớn hay nhỏ?
- Sai khác này có làm thay đổi nhãn dự đoán không?

## 5. Benchmark

| Runtime | Batch size | Mean latency (ms) | Median latency (ms) | P95 latency (ms) | Throughput (img/s) | Model size (MB) |
|---|---:|---:|---:|---:|---:|---:|
| PyTorch | 1 | ... | ... | ... | ... | ... |
| ONNXRuntime | 1 | ... | ... | ... | ... | ... |
| PyTorch | 4 | ... | ... | ... | ... | ... |
| ONNXRuntime | 4 | ... | ... | ... | ... | ... |

## 6. Phân tích kết quả

Trả lời:

1. ONNXRuntime có nhanh hơn PyTorch không?
2. Batch size ảnh hưởng thế nào đến latency và throughput?
3. Vì sao cần warm-up trước khi đo benchmark?
4. Vì sao không nên chỉ đo một lần rồi kết luận?
5. Nếu triển khai lên CPU/edge device, bạn chọn batch size nào? Vì sao?

## 7. Liên hệ triển khai thực tế

Viết 5–8 dòng:

- ONNX giúp gì trong triển khai mô hình?
- Consistency test giúp phát hiện lỗi gì?
- Benchmark giúp ra quyết định kỹ thuật như thế nào?
- Nếu cần đưa mô hình vào hệ thống Smart Campus thật, còn cần kiểm thử thêm điều gì?

## 8. Kết luận

Tóm tắt:

- Export ONNX thành công hay chưa?
- Consistency test có pass không?
- Runtime nào nhanh hơn?
- Bài học chính rút ra từ lab này là gì?
