# Báo cáo Lab 16: Cloud AI Environment Setup
**Sinh viên:** Sĩ Nguyễn Đức  
**Ngày thực hiện:** 24/04/2026  
**Phương án:** CPU Instance với LightGBM (Phương án dự phòng — Section 7)

---

## 1. Lý do sử dụng CPU thay GPU

Tài khoản AWS mới bị hạn chế quota GPU mặc định ở mức **0 vCPU** cho dòng instance G/VT (`g4dn.xlarge`). Do yêu cầu tăng quota chưa được duyệt trong thời gian làm lab, phương án chuyển sang **CPU instance `r5.2xlarge`** (8 vCPU, 32 GB RAM) được áp dụng theo hướng dẫn Section 7 của README.

Do tài khoản AWS mới chỉ cho phép **Free Tier instances**, các instance `r5.2xlarge` và `t3.large` đều bị từ chối. Benchmark thực tế được chạy trên **Bastion Host `t3.micro`** (Ubuntu 22.04) — instance duy nhất được phép khởi tạo. Kết quả vẫn phản ánh đúng hiệu năng của thuật toán LightGBM trên môi trường cloud EC2.

---

## 2. Cấu hình hạ tầng Terraform

| Thành phần | Cấu hình |
|---|---|
| CPU Node (benchmark chạy tại đây) | `t3.micro` — Ubuntu 22.04 (Free Tier) |
| Bastion Host | `t3.micro` — Ubuntu 22.04 |
| Mạng | Private VPC (`10.0.0.0/16`) |
| Load Balancer | Application Load Balancer (ALB) — port 80 → 8000 |
| NAT Gateway | 1 AZ (us-east-1) |
| Storage | 30 GB gp3 EBS |

Thay đổi so với cấu hình gốc: `instance_type` từ `g4dn.xlarge` → `t3.micro` (Free Tier), AMI từ Deep Learning AMI → Ubuntu 22.04.

---

## 3. Dataset

**Credit Card Fraud Detection** (Kaggle — ULB Machine Learning Group)

| Thông số | Giá trị |
|---|---|
| Tổng số dòng | 284,807 giao dịch |
| Số features | 30 (V1–V28 + Amount + Time) |
| Tỷ lệ gian lận | 0.173% (492 / 284,807) |
| Nguồn | [kaggle.com/datasets/mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) |

Dataset cực mất cân bằng (imbalanced) — đây là đặc trưng phổ biến của bài toán fraud detection thực tế.

---

## 4. Kết quả Benchmark

> Chạy trên EC2 `t3.micro` Ubuntu 22.04 tại AWS us-east-1 — môi trường cloud thực tế.

### 4.1 Thời gian thực thi

| Bước | Thời gian |
|---|---|
| Load dataset (284,807 dòng) | **3.23s** |
| Training LightGBM | **3.75s** |
| Tổng | **6.98s** |

### 4.2 Chất lượng mô hình

| Metric | Kết quả |
|---|---|
| AUC-ROC | **0.8980** |
| Accuracy | **0.9627** |
| F1-Score | 0.0716 |
| Precision | 0.0374 |
| Recall | **0.8367** |
| Best iteration | 25 / 500 |

### 4.3 Hiệu năng Inference

| Metric | Kết quả |
|---|---|
| Latency (1 dòng) | **0.435 ms** |
| Throughput (1000 dòng) | **625,642 rows/s** |

---

## 5. Phân tích kết quả

**AUC-ROC = 0.898** cho thấy mô hình phân biệt tốt giao dịch gian lận và bình thường, dù dataset cực mất cân bằng (chỉ 0.173% là fraud).

**Recall = 0.837** là chỉ số quan trọng nhất trong bài toán fraud detection: mô hình phát hiện được **83.7% tổng số giao dịch gian lận** — tốt cho một baseline LightGBM chưa tuning.

**Precision thấp (0.037)** là trade-off điển hình khi ưu tiên Recall cao với dataset mất cân bằng — mô hình chấp nhận nhiều false positive để không bỏ sót fraud thật.

**Training chỉ 3.75 giây** trên 227,845 dòng trên `t3.micro` (1 vCPU) cho thấy LightGBM rất hiệu quả ngay cả trên instance nhỏ nhất, không cần GPU cho bài toán tabular data.

---

## 6. So sánh CPU vs GPU cho bài toán này

| Tiêu chí | CPU `t3.micro` (thực tế) | CPU `r5.2xlarge` (mục tiêu) | GPU `g4dn.xlarge` (T4) |
|---|---|---|---|
| Chi phí/giờ | ~$0.010 (Free Tier) | ~$0.504 | ~$0.526 |
| Yêu cầu quota | Không | Không | Có (bị từ chối) |
| vCPU / RAM | 1 vCPU / 1 GB | 8 vCPU / 32 GB | 4 vCPU / 16 GB + T4 GPU |
| Training time (LGBM) | **3.75s** | ~0.88s (ước tính) | Tương đương CPU |
| Inference throughput | 625,642 rows/s | ~875,000 rows/s | Tương đương CPU |

**Kết luận:** Với bài toán gradient boosting trên tabular data, CPU hoàn toàn đủ dùng — GPU chỉ thực sự cần thiết khi chạy Deep Learning model (LLM, vLLM). `t3.micro` tuy nhỏ nhưng vẫn hoàn thành benchmark trong vòng 7 giây.

---

## 7. Ước tính chi phí AWS (1 giờ)

| Dịch vụ | Chi phí/giờ |
|---|---|
| EC2 — CPU Node / Bastion (`t3.micro` x2) | ~$0.020 |
| NAT Gateway | ~$0.045 + data |
| ALB | ~$0.008 |
| **Tổng ước tính** | **~$0.073/giờ** |

---

## 8. File đính kèm

| File | Mô tả |
|---|---|
| `terraform/main.tf` | Cấu hình Terraform đã chỉnh sửa (r5.2xlarge) |
| `benchmark.py` | Script training và inference LightGBM |
| `benchmark_result.json` | Kết quả benchmark đầy đủ dạng JSON |
