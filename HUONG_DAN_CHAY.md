# Hướng dẫn chạy Lab 16 — Từng bước từ Terraform Apply

> **Môi trường:** Windows, Anaconda base, AWS CLI đã configure, Terraform đã cài  
> **Instance:** `t3.micro` (Free Tier) — CPU Node ở Private Subnet  
> **Kết quả cuối:** benchmark LightGBM chạy trên EC2, output JSON lưu lại nộp bài

---

## Chuẩn bị (chạy 1 lần duy nhất)

Mở **Anaconda Prompt** hoặc terminal có conda, di chuyển vào thư mục project:

```bash
cd "G:\code_file\python\assignments\track2\day01\Day16-Track2-Assignment"
```

---

## Bước 1 — Deploy hạ tầng AWS

```bash
cd terraform
set TF_VAR_hf_token=dummy
terraform apply -auto-approve
```

**Đợi khoảng 10–15 phút.** Khi xong sẽ in ra:

```
Outputs:
alb_dns_name      = "ai-inference-alb-xxxx.us-east-1.elb.amazonaws.com"
bastion_public_ip = "100.54.37.119"
endpoint_url      = "http://ai-inference-alb-xxxx.../v1/completions"
gpu_private_ip    = "10.0.10.90"
```

> Ghi lại `bastion_public_ip` và `gpu_private_ip` — sẽ dùng ở các bước sau.

---

## Bước 2 — Copy benchmark.py lên Bastion Host

Mở **tab terminal mới** (vẫn ở máy local, chưa SSH), chạy:

```bash
cd "G:\code_file\python\assignments\track2\day01\Day16-Track2-Assignment"

scp -i terraform/lab-key -o StrictHostKeyChecking=no benchmark.py ubuntu@100.54.37.119:~/
```

> Thay `100.54.37.119` bằng `bastion_public_ip` của bạn nếu khác.

---

## Bước 3 — SSH thẳng vào CPU Node qua Bastion (ProxyJump)

Dùng một lệnh duy nhất, không cần SSH vào Bastion trước:

```powershell
ssh -i "G:\code_file\python\assignments\track2\day01\Day16-Track2-Assignment\terraform\lab-key" `
    -o StrictHostKeyChecking=no `
    -J ubuntu@100.54.37.119 `
    ec2-user@10.0.10.90
```

Khi thấy dấu nhắc `[ec2-user@ip-10-0-10-90 ~]$` là đã vào được CPU Node.

> **Giải thích:** `-J ubuntu@100.54.37.119` nghĩa là "nhảy qua Bastion (Ubuntu) rồi vào CPU Node (Amazon Linux 2023 — user `ec2-user`)" trong một bước duy nhất.

---

## Bước 4 — Copy benchmark.py lên CPU Node (ProxyJump)

Mở **tab terminal mới** (máy local), chạy:

```powershell
scp -i "G:\code_file\python\assignments\track2\day01\Day16-Track2-Assignment\terraform\lab-key" `
    -o StrictHostKeyChecking=no `
    -J ubuntu@100.54.37.119 `
    "G:\code_file\python\assignments\track2\day01\Day16-Track2-Assignment\benchmark.py" `
    ec2-user@10.0.10.90:~/

---

## Bước 6 — Cài đặt môi trường Python trên CPU Node

```bash
sudo dnf update -y
sudo dnf install -y python3 python3-pip
pip3 install --upgrade pip
pip3 install lightgbm scikit-learn pandas numpy kaggle
```

> Bước này mất khoảng 2–3 phút.

---

## Bước 7 — Cấu hình Kaggle API và tải dataset

```bash
mkdir -p ~/.kaggle ~/ml-benchmark

cat > ~/.kaggle/kaggle.json << 'EOF'
{"username": "snguync", "key": "KGAT_6563fc2f23d86bf1f6130fd29b451f07"}
EOF

chmod 600 ~/.kaggle/kaggle.json

kaggle datasets download -d mlg-ulb/creditcardfraud --unzip -p ~/ml-benchmark/
```

> Dataset nặng 66 MB, tải mất khoảng 1–2 phút tùy tốc độ mạng EC2.  
> Sau khi giải nén sẽ có file `~/ml-benchmark/creditcard.csv` (~144 MB, 284,807 dòng).

---

## Bước 8 — Chạy Benchmark

```bash
python3 ~/benchmark.py --csv ~/ml-benchmark/creditcard.csv
```

Kết quả sẽ in ra màn hình:

```
========== BENCHMARK RESULTS ==========
  Load / generate time:    x.xxs
  Training time:           x.xxs
  Best iteration:          xx
  AUC-ROC:                 0.xxxx
  Accuracy:                0.xxxx
  F1-Score:                0.xxxx
  Precision:               0.xxxx
  Recall:                  0.xxxx
  Inference latency (1):   x.xxx ms
  Throughput (1000 rows):  xxxxxx rows/s
=======================================
Results written to ~/ml-benchmark/benchmark_result.json
Model saved to ~/ml-benchmark/model.lgb
```

---

## Bước 9 — Copy kết quả về máy local

Thoát khỏi CPU Node, về Bastion:

```bash
exit
```

Copy file JSON từ CPU Node về Bastion:

```bash
scp -o StrictHostKeyChecking=no ec2-user@10.0.10.90:~/ml-benchmark/benchmark_result.json ~/
```

Thoát khỏi Bastion về máy local:

```bash
exit
```

Copy file JSON từ Bastion về máy local:

```bash
scp -i terraform/lab-key -o StrictHostKeyChecking=no ubuntu@100.54.37.119:~/benchmark_result.json "G:\code_file\python\assignments\track2\day01\Day16-Track2-Assignment\"
```

---

## Bước 10 — Chụp màn hình AWS Billing

1. Vào [AWS Billing Console](https://console.aws.amazon.com/billing/)
2. Chọn **Bills** hoặc **Cost Explorer** → chọn ngày hôm nay
3. Chụp màn hình thể hiện EC2 và NAT Gateway đang phát sinh chi phí

---

## Bước 11 — Dọn dẹp tài nguyên (BẮT BUỘC)

Sau khi đã chụp ảnh và lưu kết quả, **xóa toàn bộ hạ tầng** để tránh mất tiền:

```bash
cd "G:\code_file\python\assignments\track2\day01\Day16-Track2-Assignment\terraform"
terraform destroy -auto-approve
```

Đợi đến khi thấy `Destroy complete!` mới đóng terminal.

---

## Tóm tắt nhanh (Cheatsheet)

| Bước | Lệnh | Chạy ở đâu |
|---|---|---|
| 1 | `terraform apply -auto-approve` | Local — thư mục `terraform/` |
| 2 | `scp ... benchmark.py ec2-user@BASTION:~/` | Local |
| 3 | `ssh -i lab-key ec2-user@BASTION_IP` | Local |
| 4 | `scp benchmark.py ec2-user@CPU_IP:~/` | Trong Bastion SSH |
| 5 | `ssh ec2-user@CPU_IP` | Trong Bastion SSH |
| 6 | `pip3 install lightgbm ...` | Trong CPU Node SSH |
| 7 | `kaggle datasets download ...` | Trong CPU Node SSH |
| 8 | `python3 benchmark.py --csv ...` | Trong CPU Node SSH |
| 9 | `scp ... benchmark_result.json` | Bastion → Local |
| 10 | Chụp AWS Billing | Trình duyệt |
| 11 | `terraform destroy -auto-approve` | Local — thư mục `terraform/` |

---

## Thông tin hạ tầng hiện tại

| Tài nguyên | Giá trị |
|---|---|
| Bastion Public IP | `100.54.37.119` |
| CPU Node Private IP | `10.0.10.90` |
| ALB DNS | `ai-inference-alb-0dc2c2b3-1539092585.us-east-1.elb.amazonaws.com` |
| Region | `us-east-1` |
| Instance type | `t3.micro` (Free Tier — do hạn chế tài khoản AWS mới) |
| Key file | `terraform/lab-key` |