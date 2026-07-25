# CT466 — Phân Cụm Khách Hàng RFM

**Sinh viên:** Nguyễn Khánh Duy  
**MSSV:** DC22V7N506  
**Môn học:** CT466 — Niên luận ngành

---

## Giới thiệu

Pipeline phân cụm khách hàng theo chỉ số **RFM (Recency, Frequency, Monetary)** sử dụng 6 thuật toán:

| Model | Loại |
|---|---|
| K-Means | Machine Learning |
| K-Means++ | Machine Learning |
| DBSCAN | Machine Learning |
| Hierarchical (Agglomerative) | Machine Learning |
| GMM (Gaussian Mixture Model) | Machine Learning |
| PCA + K-Means | Dimensionality Reduction |
| MLP (Multi-Layer Perceptron) | Supervised (pseudo-label) |

---

## Cấu trúc thư mục

```
CT466_DC22V7N506_NguyenKhanhDuy/
├── config.py                  # Tham số cấu hình tập trung
├── main.py                    # Entry point — chạy toàn bộ pipeline
├── prepare_data.py            # Tạo CSV từ file Excel thô
├── preprocessing.py           # Tiền xử lý: log, scale, PCA
├── models/
│   ├── kmeans_model.py        # K-Means & K-Means++
│   ├── dbscan_model.py        # DBSCAN
│   ├── hierarchical_model.py  # Agglomerative Clustering
│   ├── gmm_model.py           # Gaussian Mixture Model
│   ├── autoencoder_kmeans.py  # PCA + K-Means
│   └── mlp_model.py           # MLP Classifier
├── utils/
│   └── evaluation.py          # Silhouette Score
├── data_small.xlsx            # Dữ liệu giao dịch (30 KH — demo)
├── data_large.xlsx            # Dữ liệu giao dịch đầy đủ (4338 KH)
└── customer_rfm_small.csv     # RFM đã tính (tự sinh nếu chưa có)
```

---

## Yêu cầu

- Python 3.10+
- Các thư viện: `pip install pandas numpy scikit-learn matplotlib seaborn openpyxl`

---

## Hướng dẫn chạy

### Cách 1 — Chạy trực tiếp (khuyên dùng)

```bash
python main.py
```

> `main.py` tự động tạo `customer_rfm_small.csv` từ `data_small.xlsx` nếu chưa có.

### Cách 2 — Tạo dữ liệu thủ công rồi chạy

```bash
# Tập nhỏ — 30 khách hàng (dùng cho demo / báo cáo)
python prepare_data.py --small

# Tập đầy đủ — 4338 khách hàng
python prepare_data.py

# Chạy pipeline phân cụm
python main.py
```

### Đổi tập dữ liệu

Mở [`config.py`](config.py) và sửa dòng:

```python
INPUT_FILE: str = "customer_rfm_small.csv"   # ← đổi sang "customer_rfm.csv" để dùng tập đầy đủ
```

---

## Kết quả

Sau khi chạy, file `output_results.csv` sẽ chứa nhãn phân cụm từ tất cả 7 model cho mỗi khách hàng.
