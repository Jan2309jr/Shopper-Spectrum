import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import joblib

# Load data and model
@st.cache_data
def load_data():
    df = pd.read_csv("online_retail.csv")  # Your dataset
    df.dropna(subset=['CustomerID', 'StockCode'], inplace=True)
    df['CustomerID'] = df['CustomerID'].astype(int)
    return df

@st.cache_resource
def load_kmeans_model():
    return joblib.load('kmeans_model.pkl')

df = load_data()
kmeans_model = load_kmeans_model()

# Description map
desc_map = df[['StockCode', 'Description']].drop_duplicates().set_index('StockCode')['Description'].to_dict()

# Collaborative filtering setup
@st.cache_data
def build_item_similarity(df):
    customer_item = df.pivot_table(index='CustomerID', columns='StockCode', values='Quantity', aggfunc='sum', fill_value=0)
    item_customer = customer_item.T
    similarity = cosine_similarity(item_customer)
    sim_df = pd.DataFrame(similarity, index=item_customer.index, columns=item_customer.index)
    return sim_df

item_sim_df = build_item_similarity(df)

def get_top_similar(stock_code, n=5):
    if stock_code not in item_sim_df.index:
        return []
    scores = item_sim_df[stock_code].sort_values(ascending=False)[1:n+1]
    return [(code, desc_map.get(code, "Unknown"), round(score, 2)) for code, score in scores.items()]

def predict_rfm_cluster(recency, frequency, monetary):
    if kmeans_model is None:
        raise ValueError("KMeans model not loaded properly.")
    
    rfm_input = pd.DataFrame([[recency, frequency, monetary]], columns=['Recency', 'Frequency', 'Monetary'])
    cluster = kmeans_model.predict(rfm_input)[0]
    label_map = {
        0: 'High-Value Customer',
        1: 'Regular Shopper',
        2: 'Occasional Shopper',
        3: 'At-Risk Customer'
    }
    return cluster, label_map.get(cluster, "Unknown Segment")

# --- Streamlit App UI ---
st.set_page_config(layout="wide", page_title="Customer Intelligence")

# Sidebar
st.sidebar.title("📊 Home")
page = st.sidebar.radio("Navigation", ["Clustering", "Recommendation"])

# --------------------- Page: Clustering ---------------------
if page == "Clustering":
    st.markdown("## 🧠 Customer Segmentation")

    recency = st.number_input("Recency (days since last purchase)", min_value=0, value=100)
    frequency = st.number_input("Frequency (number of purchases)", min_value=0, value=5)
    monetary = st.number_input("Monetary (total spend)", min_value=0.0, value=1000.0)

    if st.button("Predict Segment", type='primary'):
        cluster, label = predict_rfm_cluster(recency, frequency, monetary)
        st.success(f"This customer belongs to: **{label}**")
        st.write(f"Cluster ID: `{cluster}`")

# --------------------- Page: Recommendation ---------------------
elif page == "Recommendation":
    st.markdown("## 🎯 Product Recommender")

    product_name_input = st.text_input("Enter Product Name").strip().upper()

    if st.button("Recommend", type='primary'):
        matches = df[df['Description'].str.upper().str.contains(product_name_input, na=False)]
        if matches.empty:
            st.warning("No matching product found. Try a different name.")
        else:
            stock_code = matches.iloc[0]['StockCode']
            st.write(f"**Found:** {desc_map.get(stock_code)} (Code: {stock_code})")

            results = get_top_similar(stock_code)
            if results:
                st.markdown("### 🔁 Recommended Products")
                for i, (code, name, sim) in enumerate(results, start=1):
                    st.markdown(
                        f"""
                        <div style='padding: 10px; margin-bottom: 10px; border: 1px solid #ddd; border-radius: 8px; background-color: #00014D;'>
                            <b>{i}. {name}</b>  
                            <div style='font-size: 12px; color: #888;'>StockCode: {code} | Similarity: {sim}</div>
                        </div>
                        """, unsafe_allow_html=True
                    )
            else:
                st.warning("No recommendations found.")
