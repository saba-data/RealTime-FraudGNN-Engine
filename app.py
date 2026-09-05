import streamlit as st
import requests
import networkx as nx
import matplotlib.pyplot as plt

# إعدادات الصفحة
st.set_page_config(
    page_title="GraphShield AI Dashboard",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ GraphShield AI: Real-Time Fraud Network Detection")
st.markdown("---")

# تقسيم الصفحة إلى عمودين
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("إدخال تفاصيل المعاملة (Simulate Live Transaction)")
    
    trans_id = st.number_input("Transaction ID", value=3000501, step=1)
    amount = st.number_input("Transaction Amount ($)", value=2450.00, step=10.0)
    card_id = st.text_input("Card ID", value="4092")
    email_domain = st.selectbox("Email Domain", ["gmail.com", "yahoo.com", "anonymous.com", "hotmail.com"])
    
    submit_btn = st.button("فحص المعاملة عبر GNN Engine", type="primary")

with col2:
    st.subheader("نتيجة التقييم والشبكة المترابطة (Analysis & Graph)")
    
    if submit_btn:
        # إرسال الطلب لخادم الـ FastAPI
        api_url = "http://127.0.0.1:8000/predict"
        payload = {
            "transaction_id": int(trans_id),
            "amount": float(amount),
            "card_id": str(card_id),
            "email_domain": str(email_domain)
        }
        
        try:
            response = requests.post(api_url, json=payload)
            if response.status_code == 200:
                result = response.json()
                
                # عرض المؤشرات
                m1, m2, m3 = st.columns(3)
                m1.metric("Risk Level", result["risk_level"])
                m2.metric("Fraud Prob", f"{result['fraud_probability'] * 100:.1f}%")
                m3.metric("Latency", result["latency_ms"])
                
                if result["is_fraud"]:
                    st.error(f"⚠️ تنبيه احتيال! الوضع: {result['status']}")
                else:
                    st.success(f"✅ المعاملة آمنة. الوضع: {result['status']}")
                
                # رسم الـ Graph المباشر للمعاملة
                st.markdown("#### 🔗 الهيكل الشبكي للمعاملة (Graph Representation)")
                G = nx.Graph()
                t_node = f"Txn:{trans_id}"
                c_node = f"Card:{card_id}"
                e_node = f"Email:{email_domain}"
                
                G.add_edge(t_node, c_node)
                G.add_edge(t_node, e_node)
                
                fig, ax = plt.subplots(figsize=(6, 3))
                pos = nx.spring_layout(G)
                nx.draw(G, pos, with_labels=True, node_color=['#ff4b4b' if result['is_fraud'] else '#4caf50', '#1f77b4', '#1f77b4'], 
                        node_size=2000, font_color='white', font_weight='bold', ax=ax)
                st.pyplot(fig)
                
            else:
                st.error("خطأ في الاتصال بالـ API")
        except Exception as ex:
            st.error(f"تأكد من تشغيل خادم Uvicorn أولاً! الخطأ: {ex}")