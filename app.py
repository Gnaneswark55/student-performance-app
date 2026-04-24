import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.title("🎓 Student Performance Prediction System")

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, sep=';')
    st.write("Data Preview", df.head())

    df_copy = df.copy()

    # ----------- FIX 1: HANDLE MISSING VALUES -----------
    df_copy = df_copy.fillna(0)

    # ----------- FIX 2: ENCODING SAFE -----------
    le = LabelEncoder()
    for col in df_copy.columns:
        if df_copy[col].dtype == 'object':
            df_copy[col] = le.fit_transform(df_copy[col].astype(str))

    # ----------- TARGET CREATION -----------
    if 'G3' in df_copy.columns:
        df_copy['result'] = df_copy['G3'].apply(lambda x: 1 if x >= 10 else 0)
        df_copy = df_copy.drop(['G3'], axis=1)

    X = df_copy.drop('result', axis=1)
    y = df_copy['result']

    # ----------- FIX 3: FORCE NUMERIC -----------
    X = X.apply(pd.to_numeric, errors='coerce')
    X = X.fillna(0)

    # ----------- SCALING -----------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ----------- MODEL -----------
    model = RandomForestClassifier()
    model.fit(X_scaled, y)

    predictions = model.predict(X_scaled)

    # ----------- METRICS -----------
    total_students = len(predictions)
    pass_count = np.sum(predictions == 1)
    fail_count = np.sum(predictions == 0)

    pass_percent = (pass_count / total_students) * 100
    fail_percent = (fail_count / total_students) * 100

    accuracy = accuracy_score(y, predictions)

    # ----------- DASHBOARD -----------
    st.subheader("📊 Model Performance Summary")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Students", total_students)
    c2.metric("Pass", pass_count)
    c3.metric("Fail", fail_count)
    c4.metric("Accuracy", f"{accuracy*100:.2f}%")

    st.write(f"Pass Percentage: {pass_percent:.2f}%")
    st.write(f"Fail Percentage: {fail_percent:.2f}%")

    # ----------- RESULTS TABLE -----------
    df['Result'] = ['Pass' if p == 1 else 'Fail' for p in predictions]

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

    st.write("Results", df)

    # ----------- DOWNLOAD -----------
    df.to_excel("results.xlsx", index=False)

    with open("results.xlsx", "rb") as file:
        st.download_button("Download Results", file, file_name="results.xlsx")
