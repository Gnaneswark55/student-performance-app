import streamlit as st
import pandas as pd

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

st.title("🎓 Student Performance Prediction System")

# Upload file
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

# 👉 IMPORTANT: everything inside this block
if uploaded_file is not None:

    # Load data
    df = pd.read_csv(uploaded_file, sep=';')
    st.write("📊 Data Preview", df.head())

    df_copy = df.copy()

    # Encode categorical columns
    le = LabelEncoder()
    for col in df_copy.columns:
        if df_copy[col].dtype == 'object':
            df_copy[col] = le.fit_transform(df_copy[col])

    # Create target
    if 'G3' in df_copy.columns:
        df_copy['result'] = df_copy['G3'].apply(lambda x: 1 if x >= 10 else 0)
        df_copy = df_copy.drop(['G3'], axis=1)

    # Split
    X = df_copy.drop('result', axis=1)
    y = df_copy['result']

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train model
    model = RandomForestClassifier()
    model.fit(X_scaled, y)

    # Predict
    predictions = model.predict(X_scaled)

    # Add results
    df['Result'] = ['Pass' if p == 1 else 'Fail' for p in predictions]

    # Weakness detection
    def detect_weakness(row):
        weak = []
        if row.get('studytime', 0) <= 1:
            weak.append("Low Study Time")
        if row.get('failures', 0) > 0:
            weak.append("Past Failures")
        if row.get('absences', 0) > 5:
            weak.append("High Absences")
        return ", ".join(weak)

    df['Weak Areas'] = df.apply(detect_weakness, axis=1)

    # Show results
    st.write("✅ Results", df)

    # Download Excel
    df.to_excel("results.xlsx", index=False)

    with open("results.xlsx", "rb") as file:
        st.download_button("📥 Download Results", file, file_name="results.xlsx")
