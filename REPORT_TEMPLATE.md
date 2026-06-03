# CSC4005 Lab 6 Report – Export ONNX + Consistency Test + Benchmark

## 1. Thông tin nhóm

| STT | Họ tên | Mã sinh viên | Lớp |
| :-: | ------ | ------------ | --- |
|  1  | Trần Trường Giang    | 1671040009          | KHMT 16-01 |
|  2  | Nguyễn Văn Huy    | 1671040013          | KHMT 16-01 |

* Link GitHub repo: [...](https://github.com/FIT-DNU-CS-16-01/csc4005-lab6-onnx-starter-kit-khmt1601_nhom11)
* Link checkpoint sử dụng: checkpoints/best_model.pt (327.4 MB) - kết quả đã train từ bài Lab5
* Link file ONNX: 

---

## 2. Mô tả mô hình đầu vào

| Nội dung      | Giá trị                                            |
| ------------- | -------------------------------------------------- |
| Bài toán      | Smart Campus Scene Classification                  |
| Dataset       | MIT Indoor Scenes 67 subset (5 lớp)                |
| Số lớp        | 5                                                  |
| Classes       | classroom, computerroom, library, corridor, office |
| Model PyTorch | ViT-B/16 (`vit_b_16`)                              |
| Checkpoint    | `checkpoints/best_model.pt`                        |
| Image size    | 224 × 224                                          |

Mô hình được huấn luyện từ bài thực hành trước sử dụng kiến trúc Vision Transformer (ViT-B/16) để phân loại cảnh trong môi trường Smart Campus. Dữ liệu gồm 5 lớp đại diện cho các khu vực thường gặp trong trường học như phòng học, phòng máy tính, thư viện, hành lang và văn phòng.

Vision Transformer là một kiến trúc hiện đại trong lĩnh vực thị giác máy tính, áp dụng cơ chế Self-Attention để học mối quan hệ giữa các vùng ảnh. So với các mô hình CNN truyền thống, ViT có khả năng khai thác thông tin toàn cục tốt hơn và thường đạt hiệu quả cao trên các bài toán phân loại ảnh khi được huấn luyện với dữ liệu phù hợp.

---

## 3. Export ONNX

| Thông số      | Giá trị                        |
| ------------- | ------------------------------ |
| ONNX path     | `outputs/vit_smartcampus.onnx` |
| Opset         | 17                             |
| Dynamic batch | Yes                            |
| Input name    | input                          |
| Output name   | logits                         |
| Status        | exported_and_checked           |

Lệnh thực hiện:

```bash
python -m src.export_onnx \
  --checkpoint checkpoints/best_model.pt \
  --onnx_path outputs/vit_smartcampus.onnx \
  --model_name vit_b_16 \
  --img_size 224 \
  --opset 17 \
  --dynamic_batch
```

### Kết quả

Mô hình được chuyển đổi thành công sang định dạng ONNX và vượt qua bước kiểm tra tính hợp lệ bằng ONNX Checker.

Trong quá trình export, mô hình được đưa về chế độ đánh giá (`model.eval()`) nhằm đảm bảo các thành phần như Dropout hoạt động đúng trong giai đoạn suy luận.

Ngoài file `vit_smartcampus.onnx`, trọng số của mô hình được lưu trong file `.onnx.data`. Tổng dung lượng mô hình sau khi export xấp xỉ 327 MB, tương đương với checkpoint gốc của ViT-B/16.

### Vai trò của ONNX

ONNX (Open Neural Network Exchange) là định dạng trung gian cho phép trao đổi mô hình giữa nhiều framework khác nhau như PyTorch, TensorFlow, MXNet và ONNXRuntime.

Việc chuyển đổi sang ONNX mang lại nhiều lợi ích:

* Tăng khả năng tương thích giữa các nền tảng triển khai.
* Giảm phụ thuộc vào framework huấn luyện ban đầu.
* Hỗ trợ tối ưu hóa suy luận thông qua ONNXRuntime.
* Thuận tiện khi triển khai trên server, edge device hoặc môi trường production.

### Nhận xét

Việc export thành công chứng tỏ kiến trúc và trọng số mô hình đã được chuyển đổi đầy đủ sang ONNX. Tùy chọn Dynamic Batch giúp mô hình có thể xử lý nhiều kích thước batch khác nhau mà không cần export lại, tạo thuận lợi cho quá trình benchmark và triển khai thực tế.

---

## 4. Consistency Test

| Metric          | Giá trị   |
| --------------- | --------- |
| passed          | True      |
| num_samples     | 32        |
| batch_size      | 8         |
| max_abs_diff    | 3.147e-05 |
| mean_abs_diff   | 7.044e-06 |
| pred_match_rate | 1.0       |
| atol            | 1e-4      |
| rtol            | 1e-3      |

Lệnh thực hiện:

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

### Mục đích của Consistency Test

Sau khi export sang ONNX, cần kiểm tra xem mô hình mới có tạo ra kết quả tương đương với mô hình PyTorch hay không.

Consistency Test được sử dụng để:

* So sánh đầu ra logits giữa hai runtime.
* Kiểm tra độ lệch số học sau khi chuyển đổi.
* Đánh giá mức độ tương đồng của dự đoán cuối cùng.
* Phát hiện các lỗi liên quan đến export hoặc preprocessing.

### Phân tích kết quả

Kết quả cho thấy mô hình ONNX tái hiện gần như chính xác hoàn toàn hành vi của mô hình PyTorch.

* Tỷ lệ dự đoán trùng khớp đạt 100% (`pred_match_rate = 1.0`), nghĩa là toàn bộ 32 ảnh kiểm thử đều nhận cùng nhãn dự đoán ở cả hai môi trường.
* Sai số lớn nhất chỉ ở mức `3.147 × 10⁻⁵`, thấp hơn đáng kể so với ngưỡng cho phép (`1 × 10⁻⁴`).
* Sai số trung bình ở mức `7.044 × 10⁻⁶`, cho thấy chênh lệch giữa các vector logits gần như không đáng kể.
* Điều kiện kiểm tra với `atol = 1e-4` và `rtol = 1e-3` đều được thỏa mãn.

Các khác biệt rất nhỏ này chủ yếu xuất phát từ cách biểu diễn số thực và tối ưu phép toán khác nhau giữa PyTorch và ONNXRuntime, không ảnh hưởng đến kết quả phân loại cuối cùng.

### Ý nghĩa của các chỉ số

* **max_abs_diff**: Sai số tuyệt đối lớn nhất giữa hai đầu ra.
* **mean_abs_diff**: Sai số trung bình trên toàn bộ logits.
* **pred_match_rate**: Tỷ lệ dự đoán giống nhau giữa hai runtime.
* **atol** và **rtol**: Ngưỡng sai số tuyệt đối và tương đối được sử dụng để đánh giá tính tương đương.

### Kết luận

Consistency Test đạt yêu cầu và xác nhận rằng mô hình ONNX có thể thay thế mô hình PyTorch trong giai đoạn suy luận mà vẫn duy trì độ chính xác tương đương.

---

## 5. Benchmark

Benchmark được thực hiện trên CPU với:

* Warmup = 10
* Repeat = 50
* Batch size = 1, 8, 16

### Cấu hình benchmark

| Thông số          | Giá trị              |
| ----------------- | -------------------- |
| Device            | CPU                  |
| Warmup runs       | 10                   |
| Repeat runs       | 50                   |
| Batch size tested | 1, 8, 16             |
| Runtime so sánh   | PyTorch, ONNXRuntime |

Do mô hình được export với Dynamic Batch, cùng một file ONNX có thể được sử dụng để benchmark với nhiều kích thước batch khác nhau mà không cần export lại mô hình.

| Runtime     | Batch size | Mean latency (ms) | Median latency (ms) | P95 latency (ms) | Throughput (img/s) | Model size (MB) |
| ----------- | ---------- | ----------------- | ------------------- | ---------------- | ------------------ | --------------- |
| PyTorch     | 1          | 120.28            | 121.56              | 138.65           | 8.31               | 327.37          |
| ONNXRuntime | 1          | 109.25            | 106.13              | 133.97           | 9.15               | 327.37*         |
| PyTorch     | 8          | 842.31            | 835.47              | 901.62           | 9.50               | 327.37          |
| ONNXRuntime | 8          | 701.84            | 694.28              | 758.91           | 11.40              | 327.37*         |
| PyTorch     | 16         | 1635.72           | 1618.55             | 1712.84          | 9.78               | 327.37          |
| ONNXRuntime | 16         | 1328.46           | 1315.27             | 1398.63          | 12.04              | 327.37*         |

* File `.onnx` chỉ chứa graph mô hình, trong khi phần lớn trọng số được lưu trong file `.onnx.data`. Vì vậy tổng kích thước thực tế của mô hình ONNX xấp xỉ checkpoint PyTorch.

Lệnh thực hiện:

```bash
python -m src.benchmark \
  --checkpoint checkpoints/best_model.pt \
  --onnx_path outputs/vit_smartcampus.onnx \
  --data_dir data/mit_indoor_smartcampus_5 \
  --batch_sizes 1 8 16 \
  --warmup 10 \
  --repeat 50
```

### Nhận xét sơ bộ

Kết quả benchmark cho thấy ONNXRuntime duy trì lợi thế về tốc độ suy luận ở tất cả các batch size được đánh giá. Khi batch size tăng từ 1 lên 8 và 16, throughput tăng đáng kể trong khi ONNXRuntime vẫn giữ mức latency thấp hơn PyTorch.

---

## 6. Phân tích kết quả

### 1. Hiệu năng của ONNXRuntime so với PyTorch

Kết quả benchmark cho thấy ONNXRuntime hoạt động hiệu quả hơn PyTorch ở mọi kích thước batch.

* Với batch size = 1, mean latency giảm từ 120.28 ms xuống còn 109.25 ms.
* Với batch size = 8, mean latency giảm từ 842.31 ms xuống còn 701.84 ms.
* Với batch size = 16, mean latency giảm từ 1635.72 ms xuống còn 1328.46 ms.
* Throughput của ONNXRuntime luôn cao hơn PyTorch ở tất cả các trường hợp.

Điều này chứng tỏ ONNXRuntime tận dụng tốt các cơ chế tối ưu đồ thị tính toán trong giai đoạn inference.

### 2. Ý nghĩa của Latency và Throughput

Latency phản ánh thời gian cần thiết để mô hình xử lý một yêu cầu suy luận. Với các ứng dụng thời gian thực như camera giám sát hoặc nhận dạng cảnh trực tiếp, latency thấp là yếu tố quan trọng.

Throughput thể hiện số lượng ảnh được xử lý trong một giây. Chỉ số này đặc biệt hữu ích khi hệ thống cần xử lý dữ liệu liên tục hoặc nhiều luồng yêu cầu đồng thời.

Trong thí nghiệm này, ONNXRuntime đạt cả latency thấp hơn và throughput cao hơn, cho thấy hiệu quả sử dụng tài nguyên tốt hơn.

### 3. Ảnh hưởng của Dynamic Batch

Dynamic Batch cho phép mô hình nhận đầu vào với nhiều kích thước batch khác nhau mà không cần export lại.

Kết quả benchmark cho thấy:

* Batch size lớn hơn giúp tăng throughput tổng thể.
* Hiệu quả xử lý tài nguyên được cải thiện khi nhiều ảnh được suy luận cùng lúc.
* ONNXRuntime tiếp tục duy trì lợi thế khi batch size tăng.

Điều này đặc biệt hữu ích trong các hệ thống thực tế khi lưu lượng dữ liệu thay đổi theo thời gian.

### 4. Tại sao cần Warm-up?

Những lần chạy đầu tiên thường bao gồm các chi phí khởi tạo như:

* Nạp mô hình vào bộ nhớ.
* Khởi tạo runtime.
* Tối ưu graph tính toán.
* Cấp phát bộ nhớ đệm.

Nếu đưa các lần chạy này vào kết quả benchmark sẽ làm tăng độ lệch của số liệu. Vì vậy 10 lần warm-up được sử dụng để loại bỏ ảnh hưởng của giai đoạn khởi tạo.

### 5. Tại sao cần lặp lại nhiều lần?

Hiệu năng hệ thống có thể thay đổi do nhiều yếu tố như tải CPU, tiến trình nền hoặc bộ nhớ.

Việc thực hiện 50 lần đo giúp:

* Giảm ảnh hưởng của các lần chạy bất thường.
* Tính toán được Mean, Median và P95.
* Đưa ra kết quả đáng tin cậy hơn so với chỉ đo một lần duy nhất.

### 6. Lựa chọn runtime khi triển khai thực tế

Đối với bài toán phân loại cảnh trong Smart Campus, ONNXRuntime là lựa chọn phù hợp hơn vì:

* Tốc độ suy luận nhanh hơn.
* Thời gian phản hồi ổn định hơn.
* Không phụ thuộc trực tiếp vào PyTorch.
* Dễ tích hợp vào nhiều môi trường triển khai khác nhau.
* Hỗ trợ tối ưu hóa cho nhiều nền tảng phần cứng.

---

## 7. Liên hệ triển khai thực tế

Việc chuyển đổi mô hình sang ONNX giúp quá trình triển khai linh hoạt hơn vì mô hình có thể chạy trên nhiều nền tảng khác nhau mà không cần giữ nguyên môi trường huấn luyện ban đầu.

Consistency Test đóng vai trò như bước kiểm định chất lượng sau khi export. Nếu kết quả không đạt, hệ thống có thể xuất hiện các lỗi khó phát hiện như sai preprocessing, sai checkpoint hoặc sai cấu hình input.

Benchmark cung cấp cơ sở định lượng để lựa chọn giải pháp triển khai. Thay vì dựa trên cảm nhận, nhóm có thể đánh giá trực tiếp tốc độ, độ ổn định và khả năng xử lý của từng runtime.

Trong hệ thống Smart Campus, mô hình có thể được tích hợp vào:

* Camera giám sát thông minh.
* Hệ thống quản lý phòng học.
* Hệ thống hỗ trợ điều hướng trong khuôn viên trường.
* Các ứng dụng phân tích dữ liệu hình ảnh theo thời gian thực.

Trong môi trường thực tế, ngoài benchmark hiện tại cần thực hiện thêm:

* Kiểm thử với dữ liệu từ camera thật.
* Đánh giá hiệu năng khi nhiều người dùng truy cập đồng thời.
* Kiểm tra mức sử dụng CPU và RAM trong thời gian dài.
* Theo dõi độ ổn định khi hệ thống hoạt động liên tục.
* Đánh giá khả năng mở rộng khi số lượng camera tăng lên.

---

## 8. Kết luận

* Mô hình ViT-B/16 đã được export thành công sang định dạng ONNX với Opset 17 và hỗ trợ Dynamic Batch.
* ONNX Checker xác nhận mô hình ONNX hợp lệ và có thể sử dụng cho suy luận.
* Consistency Test đạt yêu cầu với `pred_match_rate = 1.0`, xác nhận mô hình ONNX và PyTorch cho kết quả tương đương.
* Sai số giữa hai môi trường suy luận rất nhỏ (`max_abs_diff = 3.147e-05`), không ảnh hưởng đến kết quả phân loại.
* Benchmark cho các batch size 1, 8 và 16 cho thấy ONNXRuntime đạt latency thấp hơn và throughput cao hơn PyTorch.
* Dynamic Batch giúp mô hình sẵn sàng cho các kịch bản triển khai với nhiều kích thước batch khác nhau trong tương lai.
* Kết quả thực nghiệm cho thấy ONNXRuntime là lựa chọn phù hợp để triển khai mô hình phân loại cảnh Smart Campus trong môi trường thực tế.
