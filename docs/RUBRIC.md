# RUBRIC_LAB6_ONNX – CSC4005 Lab 6: Export ONNX + Consistency Test + Benchmark

## 1. Thông tin chung

| Mục | Nội dung |
|---|---|
| Học phần | CSC4005 – Học sâu |
| Lab | Lab 6 |
| Chủ đề | Export ONNX, Consistency Test và Benchmark |
| Case study | Smart Campus Scene Classification |
| Model đầu vào | Vision Transformer checkpoint từ lab trước |
| Dataset | MIT Indoor Scenes 67 – subset 5 lớp |
| Tổng điểm | 10 điểm |

Lab này đánh giá năng lực đưa mô hình từ môi trường huấn luyện sang môi trường triển khai cơ bản. Sinh viên cần export mô hình PyTorch sang ONNX, kiểm thử nhất quán giữa PyTorch và ONNXRuntime, benchmark latency/throughput/model size, sau đó phân tích kết quả có căn cứ.

---

## 2. Yêu cầu đầu ra bắt buộc

Sinh viên cần nộp đầy đủ các minh chứng sau trong repo GitHub hoặc trong báo cáo:

```text
README.md hoặc báo cáo theo REPORT_TEMPLATE.md
outputs/export_report.json
outputs/consistency_report.json
outputs/benchmark_results.csv
outputs/benchmark_summary.json
File ONNX hoặc link lưu file ONNX nếu file quá lớn
Lệnh chạy tái lập được 3 bước: export, consistency test, benchmark
```

Lưu ý:

- Không commit dataset, checkpoint lớn hoặc file ONNX quá lớn lên GitHub.
- Nếu file `.pt` hoặc `.onnx` lớn, cần lưu ngoài repo và ghi link trong báo cáo.
- Benchmark phải ghi rõ batch size, warm-up, repeat và môi trường chạy.
- Consistency test phải dùng cùng preprocessing, cùng input size và đúng checkpoint.

---

## 3. Thang điểm tổng quát

| Thành phần đánh giá | Điểm |
|---|---:|
| A. Chuẩn bị checkpoint, dữ liệu và môi trường | 1.5 |
| B. Export ONNX đúng và có báo cáo export | 2.0 |
| C. Consistency test PyTorch vs ONNX | 2.0 |
| D. Benchmark latency/throughput/model size | 2.0 |
| E. Phân tích kết quả triển khai | 1.5 |
| F. Tổ chức repo, tái lập và trình bày | 1.0 |
| **Tổng** | **10** |

---

## 4. Rubric chi tiết

### A. Chuẩn bị checkpoint, dữ liệu và môi trường – 1.5 điểm

| Mức đạt | Mô tả | Điểm |
|---|---|---:|
| Tốt | Chuẩn bị đúng checkpoint từ lab ViT trước đó; dữ liệu MIT Indoor subset 5 lớp đúng cấu trúc; cài được đầy đủ thư viện; README/báo cáo nêu rõ đường dẫn checkpoint, dữ liệu và môi trường chạy. | 1.3–1.5 |
| Đạt | Chuẩn bị được checkpoint và dữ liệu; pipeline chạy được nhưng mô tả môi trường hoặc đường dẫn còn thiếu chi tiết. | 0.9–1.2 |
| Chưa đạt | Thiếu một phần dữ liệu/checkpoint hoặc phải chỉnh sửa nhiều mới chạy được; mô tả chưa rõ. | 0.4–0.8 |
| Không đạt | Không có checkpoint hợp lệ hoặc không chuẩn bị được dữ liệu/môi trường để chạy lab. | 0–0.3 |

Minh chứng cần có:

```text
checkpoints/best_model.pt hoặc link checkpoint
data/mit_indoor_smartcampus_5 hoặc đường dẫn dữ liệu
README/báo cáo mô tả cách chuẩn bị
```

---

### B. Export ONNX đúng và có báo cáo export – 2.0 điểm

| Mức đạt | Mô tả | Điểm |
|---|---|---:|
| Tốt | Export thành công mô hình PyTorch sang ONNX; dùng đúng kiến trúc, số lớp và input size; có `dynamic_batch`; file ONNX được kiểm tra bằng ONNX checker; có `export_report.json` đầy đủ. | 1.7–2.0 |
| Đạt | Export thành công ONNX; có báo cáo export nhưng còn thiếu một vài thông tin như model size, opset hoặc dynamic batch. | 1.2–1.6 |
| Chưa đạt | Export được nhưng chưa chứng minh file ONNX hợp lệ; cấu hình chưa rõ; có nguy cơ sai model/sai số lớp/sai input size. | 0.6–1.1 |
| Không đạt | Không export được ONNX hoặc file ONNX không dùng được. | 0–0.5 |

`export_report.json` nên có tối thiểu:

```text
onnx_path
checkpoint
model_name
num_classes
img_size
opset
dynamic_batch
onnx_size_mb
status
```

---

### C. Consistency test PyTorch vs ONNX – 2.0 điểm

| Mức đạt | Mô tả | Điểm |
|---|---|---:|
| Tốt | Chạy consistency test trên nhiều mẫu; so sánh logits PyTorch và ONNX; báo cáo `max_abs_diff`, `mean_abs_diff`, `pred_match_rate`; giải thích rõ pass/fail; dùng cùng preprocessing và `model.eval()`. | 1.7–2.0 |
| Đạt | Có consistency test và kết quả cơ bản; `pred_match_rate` hoặc sai khác logits được báo cáo nhưng phân tích còn ngắn. | 1.2–1.6 |
| Chưa đạt | Chỉ test 1–2 mẫu hoặc chỉ so sánh nhãn dự đoán; thiếu thông tin sai khác logits; chưa giải thích ngưỡng. | 0.6–1.1 |
| Không đạt | Không có consistency test hoặc test không chạy được. | 0–0.5 |

`consistency_report.json` nên có tối thiểu:

```text
num_samples
batch_size
max_abs_diff
mean_abs_diff
pred_match_rate
atol
rtol
passed
```

Câu hỏi cần trả lời trong báo cáo:

1. PyTorch và ONNX có cho output gần nhau không?
2. Sai khác logits có làm thay đổi nhãn dự đoán không?
3. Nếu test fail, nguyên nhân có thể nằm ở đâu?

---

### D. Benchmark latency/throughput/model size – 2.0 điểm

| Mức đạt | Mô tả | Điểm |
|---|---|---:|
| Tốt | Benchmark PyTorch và ONNXRuntime với nhiều batch size; có warm-up và repeat; báo cáo mean latency, median latency, p95 latency, throughput và model size; điều kiện benchmark rõ ràng. | 1.7–2.0 |
| Đạt | Có benchmark các metric chính; có warm-up hoặc repeat nhưng chưa đầy đủ; phân tích còn ngắn. | 1.2–1.6 |
| Chưa đạt | Benchmark sơ sài; chỉ đo một lần hoặc chỉ có latency; thiếu batch size, throughput hoặc model size. | 0.6–1.1 |
| Không đạt | Không benchmark hoặc benchmark không đáng tin cậy. | 0–0.5 |

`benchmark_results.csv` nên có các cột:

```text
runtime
batch_size
mean_latency_ms
median_latency_ms
p95_latency_ms
throughput_img_per_sec
model_size_mb
```

Cấu hình benchmark khuyến nghị:

```bash
--batch_sizes 1 4 8
--warmup 10
--repeat 50
```

---

### E. Phân tích kết quả triển khai – 1.5 điểm

| Mức đạt | Mô tả | Điểm |
|---|---|---:|
| Tốt | Phân tích rõ ONNXRuntime có nhanh hơn PyTorch không, nhanh hơn ở batch size nào, throughput thay đổi ra sao, model size có ý nghĩa gì, và runtime nào phù hợp với bối cảnh Smart Campus. | 1.3–1.5 |
| Đạt | Có nhận xét dựa trên số liệu nhưng chưa sâu; còn thiếu liên hệ với bối cảnh triển khai. | 0.9–1.2 |
| Chưa đạt | Nhận xét chung chung, chủ yếu lặp lại số liệu, chưa giải thích được ý nghĩa latency/throughput/p95. | 0.4–0.8 |
| Không đạt | Không có phân tích hoặc kết luận không dựa trên số liệu. | 0–0.3 |

Một phân tích tốt cần trả lời:

1. ONNXRuntime có nhanh hơn PyTorch không?
2. Kết quả có ổn định không? P95 latency có lệch nhiều so với mean không?
3. Batch size ảnh hưởng thế nào đến throughput?
4. Nếu triển khai thật, nên dùng runtime và batch size nào?
5. Còn cần kiểm thử thêm điều gì trước khi đưa vào hệ thống thật?

---

### F. Tổ chức repo, tái lập và trình bày – 1.0 điểm

| Mức đạt | Mô tả | Điểm |
|---|---|---:|
| Tốt | Repo sạch; không commit file lớn; README/báo cáo có lệnh chạy rõ ràng; kết quả lưu đúng thư mục; người khác có thể tái lập quy trình. | 0.9–1.0 |
| Đạt | Repo tương đối đầy đủ; có hướng dẫn chạy nhưng còn thiếu một số chi tiết nhỏ. | 0.6–0.8 |
| Chưa đạt | Repo khó theo dõi; thiếu hướng dẫn; file kết quả đặt lộn xộn hoặc thiếu tên rõ ràng. | 0.3–0.5 |
| Không đạt | Repo thiếu nhiều file chính hoặc không thể kiểm tra. | 0–0.2 |

Có thể kiểm tra nhanh:

```bash
python ci/check_structure.py
```

---

## 5. Điểm cộng khuyến khích

Sinh viên có thể được cộng tối đa **0.5 điểm khuyến khích**, nhưng tổng điểm cuối cùng không vượt quá 100.

| Nội dung cộng điểm | Điểm cộng tối đa |
|---|---:|
| Benchmark thêm nhiều batch size hoặc nhiều môi trường CPU/GPU khác nhau | +0.1 |
| Có biểu đồ latency/throughput thay vì chỉ bảng số liệu | +0.1 |
| Có phân tích chi tiết các trường hợp consistency test fail | +0.1 |
| Có so sánh ONNXRuntime với nhiều cấu hình provider/thread khác nhau | +0.1 |
| Có đề xuất bước tiếp theo sang compression/quantization cho Lab 7 | +0.1 |

---

## 6. Các lỗi bị trừ điểm mạnh

| Lỗi | Mức trừ gợi ý |
|---|---:|
| Không export được ONNX | -1.5 đến -2.5 |
| Không có consistency test | -1.5 đến -2.0 |
| Không có benchmark | -1.5 đến -2.0 |
| Benchmark không warm-up và chỉ đo một lần | -0.5 đến -1.0 |
| Dùng sai checkpoint hoặc sai số lớp | -1.0 đến -2.0 |
| Không có `export_report.json` hoặc `consistency_report.json` | -0.5 đến -1.5 |
| Không giải thích kết quả benchmark | -0.5 đến -1.5 |
| Commit dataset/checkpoint/model lớn lên GitHub | -0.5 đến -1.5 |
| Code không chạy được do lỗi import/cấu trúc | -1.0 đến -3.0 |

---

