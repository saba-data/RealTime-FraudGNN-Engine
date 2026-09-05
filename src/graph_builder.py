import torch
from torch_geometric.data import Data
import pandas as pd
from sklearn.preprocessing import LabelEncoder

class GraphBuilder:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def build_graph(self) -> Data:
        print("⏳ جاري بناء هيكل الـ Graph (Nodes & Edges)...")
        
        # 1. تجهيز المعرفات والعُقد الفريدة
        # سنربط المعاملات بناءً على card1 و P_emaildomain كـ Entities رئيسية
        self.df['card1'] = self.df['card1'].fillna(-1).astype(str)
        self.df['P_emaildomain'] = self.df['P_emaildomain'].fillna('unknown').astype(str)
        
        # إنشاء مشفرات للعقد
        card_le = LabelEncoder()
        email_le = LabelEncoder()
        
        cards = card_le.fit_transform(self.df['card1'])
        emails = email_le.fit_transform(self.df['P_emaildomain'])
        
        num_transactions = len(self.df)
        num_cards = len(card_le.classes_)
        num_emails = len(email_le.classes_)
        
        # تحويل المعرفات لإزاحات فريدة (Offset IDs)
        card_nodes = cards + num_transactions
        email_nodes = emails + num_transactions + num_cards
        
        # 2. بناء الحواف (Edges) بين المعاملة والبطاقة/الإيميل
        edge_list_src = []
        edge_list_dst = []
        
        for idx in range(num_transactions):
            # المعاملة متصلة بالبطاقة
            edge_list_src.extend([idx, card_nodes[idx]])
            edge_list_dst.extend([card_nodes[idx], idx])
            
            # المعاملة متصلة بالإيميل
            edge_list_src.extend([idx, email_nodes[idx]])
            edge_list_dst.extend([email_nodes[idx], idx])
            
        edge_index = torch.tensor([edge_list_src, edge_list_dst], dtype=torch.long)
        
        # 3. ميزات العقد (Node Features - X)
        # ميزات المعاملات الأساسية: TransactionAmt
        amounts = torch.tensor(self.df['TransactionAmt'].values, dtype=torch.float).unsqueeze(1)
        
        # إعطاء ميزات وهمية للعقد غير المباشرة (البطاقات والإيميلات)
        total_nodes = num_transactions + num_cards + num_emails
        x = torch.zeros((total_nodes, 1), dtype=torch.float)
        x[:num_transactions] = amounts
        
        # 4. التصنيف المستهدف (Labels - Y)
        y = torch.zeros(total_nodes, dtype=torch.long)
        y[:num_transactions] = torch.tensor(self.df['isFraud'].values, dtype=torch.long)
        
        # 5. بناء قناع التدريب والحر (Train / Test Masks)
        train_mask = torch.zeros(total_nodes, dtype=torch.bool)
        test_mask = torch.zeros(total_nodes, dtype=torch.bool)
        
        split_idx = int(num_transactions * 0.8)
        train_mask[:split_idx] = True
        test_mask[split_idx:num_transactions] = True
        
        # تجميع البيانات في PyG Data Object
        data = Data(x=x, edge_index=edge_index, y=y)
        data.train_mask = train_mask
        data.test_mask = test_mask
        
        print(f"✅ تم بناء الـ Graph بنجاح!")
        print(f"📊 إجمالي العُقد (Nodes): {data.num_nodes}")
        print(f"🔗 إجمالي الروابط (Edges): {data.num_edges}")
        return data

if __name__ == "__main__":
    from data_loader import IEEEDataLoader
    
    loader = IEEEDataLoader()
    df = loader.load_data()
    
    builder = GraphBuilder(df)
    graph_data = builder.build_graph()