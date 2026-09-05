import os
import sys
import time
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from torch_geometric.data import Data

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models import FraudGNN

app = FastAPI(
    title="GraphShield AI - Real-Time Fraud Detection Engine",
    description="Inference API using Graph Neural Networks (GNN) for Real-Time Fraud Detection",
    version="1.0.0"
)

MODEL_PATH = "fraud_gnn_model.pth"
model = FraudGNN(in_channels=1, hidden_channels=32, out_channels=2)

if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()
    print("✅ تم تحميل نموذج الـ Fraud GNN بنجاح!")
else:
    print("⚠️ ملف النموذج غير موجود.")

class TransactionRequest(BaseModel):
    transaction_id: int
    amount: float
    card_id: str
    email_domain: str

@app.post("/predict")
def predict_fraud(transaction: TransactionRequest):
    start_time = time.time()

    try:
        # 1. تحجيم المبلغ (Normalization) ليطابق مجال ميزات التدريب
        scaled_amount = transaction.amount / 1000.0
        
        # 2. تجهيز عقد ميزات المعاملات والمحيط الشبكي
        x = torch.tensor([[scaled_amount], [0.1], [0.1]], dtype=torch.float)
        edge_index = torch.tensor([
            [0, 1, 0, 2],
            [1, 0, 2, 0]
        ], dtype=torch.long)

        # 3. التمرير عبر الـ GNN
        with torch.no_grad():
            out = model(x, edge_index)
            probabilities = F.softmax(out, dim=1)
            raw_prob = probabilities[0][1].item()

        # 4. دمج إشارات الخطر الشبكية (Heuristic & Topology Risk Integration)
        # ميزات عالية الخطر: المبالغ الكبيرة (> 3000) أو الإيميلات المجهولة
        risk_score = raw_prob
        if transaction.amount > 3000.0 or transaction.email_domain == "anonymous.com":
            risk_score = min(0.98, max(0.82, raw_prob + 0.80))
        else:
            risk_score = round(raw_prob, 4)

        execution_time_ms = round((time.time() - start_time) * 1000, 2)
        is_fraud = risk_score > 0.50
        risk_level = "HIGH" if risk_score > 0.70 else ("MEDIUM" if risk_score > 0.35 else "LOW")

        return {
            "transaction_id": transaction.transaction_id,
            "fraud_probability": round(risk_score, 4),
            "is_fraud": is_fraud,
            "risk_level": risk_level,
            "latency_ms": f"{execution_time_ms} ms",
            "status": "FLAGGED_FOR_REVIEW" if is_fraud else "APPROVED"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def root():
    return {"message": "GraphShield AI Engine is Online and Ready."}