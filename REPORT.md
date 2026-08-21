# BÁO CÁO THỰC HÀNH MLOPS: TỪ THỰC NGHIỆM CỤC BỘ ĐẾN TRIỂN KHAI LIÊN TỤC

**Khóa học**: AI In Action - VinUni  
**Buổi**: Day 21 - CI/CD cho AI Systems (Track 2)  
**Học viên**: Trần Duy Trường  
**Repository**: https://github.com/truong311203-arch/TRACK2-DAY21-2A202601247-TranDuyTruong  

---

## 1. Kết Quả Bước 1: Thực Nghiệm Cục Bộ & Theo Dõi Bằng MLflow

- **Dataset**: Wine Quality (UCI Machine Learning Repository) gồm 12 đặc trưng hóa học, phân loại chất lượng rượu vào 3 lớp (`0`: Thấp, `1`: Trung bình, `2`: Cao).
- **Backend lưu trữ**: SQLite (`sqlite:///mlflow.db`) kết hợp Artifact storage (`./mlartifacts`).
- **Các thí nghiệm đã thực hiện**:

| Run ID | Thuật toán | `n_estimators` | `max_depth` | `min_samples_split` | Accuracy | F1-score (weighted) |
|---|---|---|---|---|---|---|
| `7d47de38` | `RandomForestClassifier` | 100 | 5 | 2 | 0.5640 | 0.5534 |
| `9d2209bb` | `RandomForestClassifier` | 50 | 3 | 2 | 0.5580 | 0.5185 |
| `0ecb3bc3` | `RandomForestClassifier` | 200 | 15 | 2 | 0.6640 | 0.6620 |
| `e90eb08b` | `RandomForestClassifier` | 200 | 20 | 2 | **0.6840** | **0.6830** |

### Giải thích lựa chọn siêu tham số tốt nhất:
Bộ tham số `n_estimators: 200`, `max_depth: 20`, `min_samples_split: 2` cho kết quả cao nhất nhờ việc tăng độ sâu cây quyết định giúp mô hình học được các tương tác phi tuyến tính phức tạp giữa 12 chỉ số hóa học (nồng độ cồn, độ pH, sunphat...) mà không bị underfitting như các cây nông (`max_depth: 3-5`).

---

## 2. Kết Quả Bước 2: Pipeline CI/CD Tự Động Với DVC & AWS (S3 + EC2)

- **Cloud Object Storage**: AWS S3 Bucket `mlops-wine-quality-tdt31` (quản lý qua DVC).
- **Cloud VM**: AWS EC2 Instance (`13.236.9.99`, Amazon Linux 2023).
- **Inference Server**: FastAPI chạy dưới dạng Systemd service `mlops-serve.service`.
- **Pipeline GitHub Actions**: Gồm 4 jobs tuần tự:
  1. **Job 1 (Unit Test)**: Chạy `pytest tests/ -v` trên dữ liệu synthetic (3/3 passed).
  2. **Job 2 (Train)**: Xác thực AWS, kéo dữ liệu từ S3 qua `dvc pull`, huấn luyện mô hình, upload `model.pkl` lên S3 và lưu artifact `outputs/metrics.json`.
  3. **Job 3 (Eval Gate)**: Đánh giá chất lượng mô hình.
  4. **Job 4 (Deploy)**: SSH vào EC2, tự động restart service `mlops-serve` và kiểm tra health check.

### Minh chứng kiểm tra API trên EC2:
- **Health Check (`GET /health`)**:
  ```powershell
  curl.exe http://13.236.9.99:8000/health
  # Trả về: {"status":"ok"}
  ```
- **Inference (`POST /predict`)**:
  ```powershell
  curl.exe --% -X POST http://13.236.9.99:8000/predict -H "Content-Type: application/json" -d "{\"features\": [7.4, 0.70, 0.00, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4, 0]}"
  # Trả về: {"prediction":0,"label":"thap"}
  ```

---

## 3. Kết Quả Bước 3: Huấn Luyện Liên Tục (Continuous Training)

- **Kịch bản**: Thêm 2998 mẫu từ `train_phase2.csv` vào `train_phase1.csv` (nâng tổng số mẫu lên 5996).
- **Tự động hóa**: Chạy `dvc add data/train_phase1.csv`, `dvc push` và commit file `data/train_phase1.csv.dvc` lên GitHub. Pipeline tự động kích hoạt bởi commit dữ liệu mới.
- **Bảng so sánh hiệu năng**:

| Giai đoạn | Số lượng mẫu huấn luyện | Accuracy | F1-score (weighted) |
|---|---|---|---|
| **Bước 2 (train_phase1)** | 2998 mẫu | 0.6840 | 0.6830 |
| **Bước 3 (train_phase1 + phase2)** | 5996 mẫu | **0.7560** | **0.7552** |

> **Nhận xét**: Khi tăng gấp đôi lượng dữ liệu huấn luyện, độ chính xác tăng từ **68.40% lên 75.60%** (vượt xa ngưỡng yêu cầu 0.70). Mô hình mới được tự động đóng gói và triển khai lên EC2 mà không cần bất kỳ can thiệp thủ công nào.

---

## 4. Khó Khăn Gặp Phải & Cách Giải Quyết

1. **Khác biệt cú pháp Availability Zone trên AWS**: Khi nhập secret region `ap-southeast-2b` (tên zone), S3 API bị lỗi kết nối do endpoint chỉ chấp nhận `ap-southeast-2` (tên region).  
   *Giải pháp*: Đã bổ sung logic regex tự động chuẩn hóa chuỗi region trong pipeline CI/CD trước khi khởi tạo Boto3 client.
2. **Xử lý chuỗi JSON trên Windows PowerShell**: Khi gửi lệnh `curl` có chứa body JSON với dấu ngoặc kép `\"`, PowerShell mặc định bóc tách dấu quote gây lỗi cú pháp JSON.  
   *Giải pháp*: Sử dụng cờ `--%` (stop parsing) hoặc `Invoke-RestMethod` để truyền nguyên vẹn payload sang server.
