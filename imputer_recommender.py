import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Imputation Method Recommender", layout="centered")

st.title("Imputation Method Recommender")
st.markdown("Upload a CSV dataset and receive a recommendation for the best imputation technique based on its features.")

uploaded_file = st.file_uploader("Upload your CSV dataset", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # To analyze dataset characteristics
    num_features = df.select_dtypes(include=[np.number]).shape[1]
    cat_features = df.select_dtypes(exclude=[np.number]).shape[1]
    missing_rate = df.isnull().mean().mean()

    st.subheader("📊 Dataset Summary")
    st.write({
        "Samples": df.shape[0],
        "Total Features": df.shape[1],
        "Numerical Features": num_features,
        "Categorical Features": cat_features,
        "Missing Rate (%)": round(missing_rate * 100, 2)
    })

    # Recommendation engine
    def recommend_method(missing, num_feat, cat_feat):
        if missing < 0.05:
            return "KNN or Mean — work well for low missingness."
        elif cat_feat > num_feat:
            return "MICE — good for categorical-rich datasets."
        elif num_feat > 20:
            return "Autoencoder — good performance for complex, high-dimensional data."
        else:
            return "MICE or KNN — maintains performance in most clinical datasets."

    recommendation = recommend_method(missing_rate, num_features, cat_features)

    st.success("Recommended Imputation Method:")
    st.write(recommendation)
