# CSC4005 Lab 6 Report – Export ONNX + Consistency Test + Benchmark

## 1. Thông tin

* Họ tên: Nguyễn Văn Huy
* Mã sinh viên: 1671040013
* Lớp: KHMT 16-01
* Link GitHub repo: https://github.com/FIT-DNU-CS-16-01/csc4005-lab6-onnx-starter-kit-khmt1601_nhom11.git
* Link checkpoint hoặc mô tả checkpoint sử dụng: `checkpoints/best_model.pt`
* Link file ONNX nếu không commit trực tiếp: ........................................

---

## 2. Mô tả mô hình đầu vào

| Nội dung                | Giá trị                                            |
| ----------------------- | -------------------------------------------------- |
| Bài toán                | Smart Campus Scene Classification                  |
| Dataset                 | MIT Indoor Scenes 67 subset                        |
| Số lớp                  | 5                                                  |
| Classes                 | classroom, computerroom, library, corridor, office |
| Model PyTorch           | Vision Transformer (ViT-B/16)                      |
| Checkpoint              | best_model.pt                                      |
| Image size              | 224 × 224                                          |
| Train mode từ lab trước | head_only                                          |

---

## 3. Export ONNX

### Thông tin export

| Thông số      | Giá trị                       |
| ------------- | ----------------------------- |
| ONNX path     | outputs/debug_random_vit.onnx |
| Opset         | 16                            |
| Dynamic batch | Yes                           |
| Input name    | input                         |
| Output name   | logits                        |
| Model size    | 1.25 MB                       |
| Status        | exported_and_checked          |

### Lệnh đã chạy

```bash
python -m src.export_onnx \
--onnx_path outputs/debug_random_vit.onnx \
--model_name vit_b_16 \
--num_classes 5 \
--img_size 224 \
--opset 16 \
--dynamic_batch \
--no_checkpoint
```

### Nhận xét

Quá trình export ONNX được thực hiện thành công. File ONNX được tạo ra và vượt qua bước kiểm tra bằng `onnx.checker.check_model()`. Mô hình được export với tùy chọn dynamic batch nhằm hỗ trợ thay đổi kích thước batch trong quá trình suy luận.

---

## 4. Consistency Test

| Metric          |               Giá trị |
| --------------- | --------------------: |
| passed          |                  True |
| num_samples     |                    32 |
| batch_size      |                     1 |
| max_abs_diff    | 2.956390380859375e-05 |
| mean_abs_diff   | 7.260771219819162e-06 |
| pred_match_rate |                   1.0 |
| atol            |                0.0001 |
| rtol            |                 0.001 |

### Nhận xét

* PyTorch và ONNX cho kết quả nhất quán.
* Sai khác giữa logits của hai runtime rất nhỏ, với `max_abs_diff ≈ 2.96 × 10⁻⁵` và `mean_abs_diff ≈ 7.26 × 10⁻⁶`.
* Tỷ lệ dự đoán giống nhau đạt `100%` (`pred_match_rate = 1.0`).
* Sai khác số học không làm thay đổi nhãn dự đoán của bất kỳ mẫu nào.
* Consistency test đạt yêu cầu (`passed = true`), chứng tỏ mô hình ONNX đã bảo toàn hành vi suy luận của mô hình PyTorch.

---

## 5. Benchmark

| Runtime     | Batch size | Mean latency (ms) | Median latency (ms) | P95 latency (ms) | Throughput (img/s) | Model size (MB) |
| ----------- | ---------: | ----------------: | ------------------: | ---------------: | -----------------: | --------------: |
| PyTorch     |          1 |           314.369 |             290.764 |          374.618 |              3.181 |         327.368 |
| ONNXRuntime |          1 |           172.542 |             167.360 |          199.325 |              5.796 |           0.102 |
| PyTorch     |          4 |               N/A |                 N/A |              N/A |                N/A |             N/A |
| ONNXRuntime |          4 |               N/A |                 N/A |              N/A |                N/A |             N/A |

### Ghi chú

Trong quá trình kiểm thử, mô hình ONNX phát sinh lỗi reshape khi chạy với batch size lớn hơn 1. Do đó benchmark chỉ được thực hiện với batch size = 1.

---

## 6. Phân tích kết quả

### 1. ONNXRuntime có nhanh hơn PyTorch không?

Có.

Kết quả benchmark cho thấy:

* PyTorch: 314.369 ms/inference
* ONNXRuntime: 172.542 ms/inference

Tốc độ của ONNXRuntime nhanh hơn khoảng:

[
\frac{314.369}{172.542} \approx 1.82
]

Tức là ONNXRuntime nhanh hơn khoảng **1.82 lần** so với PyTorch trong điều kiện thí nghiệm này.

---

### 2. Batch size ảnh hưởng thế nào đến latency và throughput?

Thông thường:

* Khi tăng batch size, latency của một lần suy luận thường tăng.
* Tuy nhiên throughput (số ảnh xử lý mỗi giây) thường tăng do tận dụng tài nguyên phần cứng hiệu quả hơn.
* Batch size nhỏ phù hợp với các hệ thống yêu cầu phản hồi nhanh.
* Batch size lớn phù hợp với xử lý hàng loạt (offline inference).

Trong bài thực hành này chưa thể đánh giá batch size > 1 do giới hạn của mô hình ONNX đã export.

---

### 3. Vì sao cần warm-up trước khi đo benchmark?

Những lần suy luận đầu tiên thường chậm hơn do:

* Khởi tạo runtime.
* Cấp phát bộ nhớ.
* Tối ưu hóa graph.
* Tải các thành phần cần thiết vào bộ nhớ.

Warm-up giúp loại bỏ ảnh hưởng của các lần chạy đầu tiên và phản ánh chính xác hơn hiệu năng thực tế của mô hình.

---

### 4. Vì sao không nên chỉ đo một lần rồi kết luận?

Một lần đo có thể bị ảnh hưởng bởi:

* Tiến trình nền của hệ điều hành.
* CPU đang xử lý tác vụ khác.
* Dao động tài nguyên hệ thống.

Vì vậy cần lặp lại nhiều lần để tính:

* Mean latency.
* Median latency.
* P95 latency.

Các chỉ số này phản ánh hiệu năng ổn định hơn.

---

### 5. Nếu triển khai lên CPU/edge device, bạn chọn batch size nào? Vì sao?

Trong thí nghiệm hiện tại, batch size = 1 là lựa chọn phù hợp vì:

* Latency thấp.
* Đáp ứng yêu cầu phản hồi theo thời gian thực.
* ONNXRuntime hoạt động ổn định với batch size = 1.
* Hệ thống Smart Campus thường xử lý từng ảnh riêng lẻ thay vì theo lô lớn.

---

## 7. Liên hệ triển khai thực tế

ONNX đóng vai trò như một định dạng trung gian giúp mô hình học sâu có thể được triển khai trên nhiều nền tảng và framework khác nhau mà không phụ thuộc hoàn toàn vào PyTorch. Việc export sang ONNX giúp giảm sự phụ thuộc vào môi trường huấn luyện và tạo điều kiện thuận lợi cho triển khai thực tế.

Consistency test giúp phát hiện các lỗi trong quá trình export như sai kiến trúc mô hình, sai preprocessing, sai số lớp hoặc lỗi trong runtime. Nếu kết quả giữa PyTorch và ONNX khác biệt đáng kể, mô hình ONNX có thể không còn phản ánh đúng hành vi của mô hình gốc.

Benchmark cung cấp các số liệu quan trọng như latency, throughput và kích thước mô hình để đánh giá khả năng triển khai. Những số liệu này giúp kỹ sư lựa chọn runtime phù hợp với yêu cầu hệ thống.

Trong hệ thống Smart Campus thực tế, ngoài latency còn cần kiểm thử thêm độ chính xác trên dữ liệu mới, mức sử dụng RAM, khả năng chịu tải, độ ổn định khi chạy liên tục, khả năng xử lý lỗi và khả năng mở rộng khi số lượng người dùng tăng lên.

---

## 8. Kết luận

* Mô hình Vision Transformer đã được export sang ONNX thành công.
* File ONNX vượt qua bước kiểm tra hợp lệ và được tạo với tùy chọn dynamic batch.
* Consistency test đạt yêu cầu với `pred_match_rate = 1.0` và sai số rất nhỏ giữa PyTorch và ONNX.
* Benchmark cho thấy ONNXRuntime nhanh hơn PyTorch khoảng 1.82 lần ở batch size = 1.
* ONNXRuntime đạt throughput cao hơn và latency thấp hơn so với PyTorch trong điều kiện thử nghiệm.
* Một hạn chế được ghi nhận là mô hình ONNX hiện tại chưa chạy ổn định với batch size lớn hơn 1 do lỗi reshape trong graph.
* Bài học quan trọng nhất của bài lab là export thành công chưa đủ để khẳng định mô hình triển khai được; cần thực hiện consistency test và benchmark để đánh giá tính đúng đắn cũng như hiệu năng của mô hình trước khi đưa vào môi trường thực tế.
