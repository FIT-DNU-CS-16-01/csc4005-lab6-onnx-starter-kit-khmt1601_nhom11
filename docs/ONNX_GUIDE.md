# ONNX Guide

## 1. ONNX là gì?

ONNX là định dạng trung gian giúp mô hình học sâu có thể được triển khai độc lập hơn với framework huấn luyện ban đầu.

Trong lab này:

```text
PyTorch model → ONNX model → ONNXRuntime inference
```

## 2. Các khái niệm chính

| Khái niệm | Ý nghĩa |
|---|---|
| opset | Phiên bản tập toán tử ONNX |
| input_names | Tên input node |
| output_names | Tên output node |
| dynamic_axes | Cho phép batch size thay đổi |
| onnxruntime | Runtime dùng để chạy inference với file ONNX |

## 3. Vì sao cần `model.eval()`?

Khi export, mô hình cần ở chế độ inference. Nếu không gọi `eval()`, các lớp như dropout hoặc batch normalization có thể hoạt động khác, làm output PyTorch và ONNX lệch nhau.

## 4. Vì sao cần dynamic batch?

Nếu không khai báo dynamic batch, file ONNX có thể chỉ nhận đúng batch size lúc export. Với triển khai thật, batch size thường thay đổi, nên nên dùng dynamic batch.
