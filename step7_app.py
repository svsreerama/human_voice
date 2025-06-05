import pandas as pd
import joblib
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Load model and scaler
model = joblib.load("voice_model.pkl")
scaler = joblib.load("scaler.pkl")

# App UI
st.set_page_config(page_title="Voice Gender Classifier", page_icon="🎙️")
st.title("🎙️ Human Voice Gender Classifier")
file = st.file_uploader("Upload CSV with audio features", type="csv")

if file:
    df = pd.read_csv(file)
    X_scaled = scaler.transform(df)
    preds = model.predict(X_scaled)

    st.write("Predictions (0=Female, 1=Male):")
    st.write(pd.DataFrame(preds, columns=["Predicted Gender"]))

    # Summary
    unique, counts = np.unique(preds, return_counts=True)
    summary = {str(int(k)): int(v) for k, v in zip(unique, counts)}
    st.write("Summary:", summary)

    # Pie Chart
    fig, ax = plt.subplots()
    ax.pie(counts, labels=["Female", "Male"], autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    st.pyplot(fig)

