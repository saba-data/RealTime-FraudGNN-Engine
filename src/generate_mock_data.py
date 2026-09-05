import os
import numpy as np
import pandas as pd

def generate_mock_ieee_data(num_samples=5000):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_dir = os.path.join(project_root, "data", "raw")
    os.makedirs(raw_dir, exist_ok=True)

    np.random.seed(42)
    trans_ids = np.arange(3000000, 3000000 + num_samples)

    # 1. إنشاء بيانات المعاملات Mock Transactions
    df_trans = pd.DataFrame({
        'TransactionID': trans_ids,
        'isFraud': np.random.choice([0, 1], size=num_samples, p=[0.96, 0.04]),
        'TransactionDT': np.random.randint(86400, 86400*30, size=num_samples),
        'TransactionAmt': np.round(np.random.exponential(scale=100, size=num_samples), 2),
        'ProductCD': np.random.choice(['W', 'H', 'C', 'S', 'R'], size=num_samples),
        'card1': np.random.randint(1000, 9999, size=num_samples),
        'card2': np.random.randint(100, 600, size=num_samples),
        'addr1': np.random.randint(100, 500, size=num_samples),
        'P_emaildomain': np.random.choice(['gmail.com', 'yahoo.com', 'hotmail.com', 'anonymous.com'], size=num_samples)
    })

    # 2. إنشاء بيانات الهوية Mock Identity (لنسبة من المعاملات)
    id_samples = int(num_samples * 0.3)
    id_trans_ids = np.random.choice(trans_ids, size=id_samples, replace=False)

    df_id = pd.DataFrame({
        'TransactionID': id_trans_ids,
        'DeviceType': np.random.choice(['desktop', 'mobile'], size=id_samples),
        'DeviceInfo': np.random.choice(['Windows', 'iOS Device', 'MacOS', 'Trident/7.0'], size=id_samples),
        'id_30': np.random.choice(['Windows 10', 'iOS 11.4.1', 'Mac OS X', 'Android 8.0'], size=id_samples),
        'id_31': np.random.choice(['chrome 66.0', 'mobile safari 11.0', 'ie 11.0'], size=id_samples)
    })

    # حفظ الملفات
    trans_path = os.path.join(raw_dir, "train_transaction.csv")
    id_path = os.path.join(raw_dir, "train_identity.csv")

    df_trans.to_csv(trans_path, index=False)
    df_id.to_csv(id_path, index=False)

    print(f"✅ تم إنشاء بيانات تجريبية بنجاح!")
    print(f"📍 Transactions: {trans_path}")
    print(f"📍 Identity: {id_path}")

if __name__ == "__main__":
    generate_mock_ieee_data()