import os
import sys
import argparse
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')

def prepare_dataset(input_xlsx: str, output_csv: str) -> None:
    
    print(f"Đang đọc '{input_xlsx}', vui lòng chờ...")
    df = pd.read_excel(input_xlsx)

    print("Đang làm sạch dữ liệu và tính toán RFM...")
    df = df.dropna(subset=['CustomerID'])
    df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
    df['CustomerID'] = df['CustomerID'].astype(int)
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

    max_date = df['InvoiceDate'].max()
    reference_date = max_date + pd.Timedelta(days=1)

    rfm = df.groupby('CustomerID').agg(
        Recency=('InvoiceDate', lambda x: (reference_date - x.max()).days),
        Frequency=('InvoiceNo', 'nunique'),
        Monetary=('TotalPrice', 'sum'),
    ).reset_index()

    out_dir = os.path.dirname(output_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    rfm.to_csv(output_csv, index=False)
    print(f"Đã lưu {len(rfm)} khách hàng vào '{output_csv}'.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tạo file CSV chứa chỉ số RFM từ dữ liệu giao dịch Excel."
    )
    parser.add_argument(
        "--small",
        action="store_true",
        help="Dùng tập nhỏ: input/data_small.xlsx → input/customer_rfm_small.csv",
    )
    args = parser.parse_args()

    os.makedirs("input", exist_ok=True)
    if args.small:
        prepare_dataset(
            input_xlsx=os.path.join("input", "data_small.xlsx"),
            output_csv=os.path.join("input", "customer_rfm_small.csv"),
        )
    else:
        prepare_dataset(
            input_xlsx=os.path.join("input", "data_large.xlsx"),
            output_csv=os.path.join("input", "customer_rfm.csv"),
        )


if __name__ == "__main__":
    main()
