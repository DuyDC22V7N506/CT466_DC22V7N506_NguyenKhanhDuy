"""
prepare_data.py
---------------
Tiền xử lý dữ liệu thô từ file Excel, tính toán chỉ số RFM
và lưu ra file CSV để dùng làm đầu vào cho pipeline phân cụm.

Hỗ trợ hai chế độ:
    python prepare_data.py            → xử lý data_large.xlsx → customer_rfm.csv
    python prepare_data.py --small    → xử lý data_small.xlsx  → customer_rfm_small.csv
"""

import sys
import argparse
import pandas as pd

# Đảm bảo stdout dùng UTF-8 trên Windows
sys.stdout.reconfigure(encoding='utf-8')


def prepare_dataset(input_xlsx: str, output_csv: str) -> None:
    """Đọc file Excel thô, làm sạch dữ liệu và tính chỉ số RFM.

    Args:
        input_xlsx: Đường dẫn file Excel đầu vào (data_large.xlsx / data_small.xlsx).
        output_csv: Đường dẫn file CSV đầu ra chứa RFM đã tính.
    """
    print(f"Đang đọc '{input_xlsx}', vui lòng chờ...")
    df = pd.read_excel(input_xlsx)

    print("Đang làm sạch dữ liệu và tính toán RFM...")
    # Bỏ CustomerID trống, Quantity và UnitPrice phải lớn hơn 0
    df = df.dropna(subset=['CustomerID'])
    df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
    df['CustomerID'] = df['CustomerID'].astype(int)
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

    # Gom nhóm theo khách hàng để tính RFM
    max_date = df['InvoiceDate'].max()
    reference_date = max_date + pd.Timedelta(days=1)

    rfm = df.groupby('CustomerID').agg(
        Recency=('InvoiceDate', lambda x: (reference_date - x.max()).days),
        Frequency=('InvoiceNo', 'nunique'),
        Monetary=('TotalPrice', 'sum'),
    ).reset_index()

    # Lưu kết quả
    rfm.to_csv(output_csv, index=False)
    print(f"Đã lưu {len(rfm)} khách hàng vào '{output_csv}'.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tạo file CSV chứa chỉ số RFM từ dữ liệu giao dịch Excel."
    )
    parser.add_argument(
        "--small",
        action="store_true",
        help="Dùng tập nhỏ: data_small.xlsx → customer_rfm_small.csv",
    )
    args = parser.parse_args()

    if args.small:
        prepare_dataset(
            input_xlsx="data_small.xlsx",
            output_csv="customer_rfm_small.csv",
        )
    else:
        prepare_dataset(
            input_xlsx="data_large.xlsx",
            output_csv="customer_rfm.csv",
        )


if __name__ == "__main__":
    main()
