import os
import pandas as pd

class IEEEDataLoader:
    def __init__(self, raw_data_path: str = None):
        if raw_data_path is None:
            # تحديد مسار المجلد الرئيسي للمشروع تلقائياً
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.raw_data_path = os.path.join(project_root, "data", "raw")
        else:
            self.raw_data_path = raw_data_path

        self.trans_path = os.path.join(self.raw_data_path, "train_transaction.csv")
        self.id_path = os.path.join(self.raw_data_path, "train_identity.csv")

    def load_data(self, nrows: int = None):
        """
        تحميل بيانات المعاملات والهويات ودمجهما
        """
        if not os.path.exists(self.trans_path):
            raise FileNotFoundError(
                f"❌ الملف غير موجود في المسار: {self.trans_path}\n"
                f"يرجى التأكد من تنزيل ملفات IEEE-CIS ووضعها داخل مجلد data/raw/"
            )

        print("⏳ جاري تحميل بيانات المعاملات (Transactions)...")
        df_trans = pd.read_csv(self.trans_path, nrows=nrows)

        print("⏳ جاري تحميل بيانات الهوية (Identity)...")
        df_id = pd.read_csv(self.id_path, nrows=nrows)

        print("🔗 جاري دمج البيانات بناءً على TransactionID...")
        df_merged = pd.merge(df_trans, df_id, on="TransactionID", how="left")
        
        print(f"✅ تم التحميل بنجاح! إجمالي الصفوف: {len(df_merged)}")
        return df_merged

if __name__ == "__main__":
    loader = IEEEDataLoader()
    df = loader.load_data(nrows=10000)
    
    print("\n--- معاينة البيانات المدمجة ---")
    print(f"عدد الأعمدة: {df.shape[1]}")
    if 'isFraud' in df.columns:
        fraud_rate = df['isFraud'].value_counts(normalize=True).get(1, 0) * 100
        print(f"نسبة حالات الاحتيال (isFraud): {fraud_rate:.2f}%")