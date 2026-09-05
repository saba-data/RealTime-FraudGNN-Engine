import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

class FraudGNN(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int):
        super(FraudGNN, self).__init__()
        # طبقتين من GraphSAGE لالتقاط العلاقات الشبكية المباشرة وغير المباشرة
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index)
        return x

def train_model(data, epochs=20):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ التجهيز للتدريب على جهاز: {device}")
    
    model = FraudGNN(in_channels=data.num_features, hidden_channels=32, out_channels=2).to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    
    model.train()
    print("🚀 بدء عملية تدريب نموذج الـ GNN...")
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        
        # حساب الخسارة فقط على عقد التدريب
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:02d}/{epochs:02d} | Loss: {loss.item():.4f}")
            
    # تقييم أداء النموذج
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    
    correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
    acc = int(correct) / int(data.test_mask.sum())
    print(f"\n🎯 دقة التنبؤ للنموذج (Accuracy on Test Set): {acc * 100:.2f}%")
    
    # حفظ وزن الموديل لاستخدامه في الـ API
    torch.save(model.state_dict(), "fraud_gnn_model.pth")
    print("💾 تم حفظ وزن الموديل في الملف: fraud_gnn_model.pth")
    return model

if __name__ == "__main__":
    from data_loader import IEEEDataLoader
    from graph_builder import GraphBuilder
    
    loader = IEEEDataLoader()
    df = loader.load_data()
    
    builder = GraphBuilder(df)
    graph_data = builder.build_graph()
    
    train_model(graph_data, epochs=20)