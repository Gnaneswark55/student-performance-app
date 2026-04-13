import streamlit as st
import pandas as pd

st.title("🎓 Student Performance Prediction System")

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file:
    # ✅ FIRST load data
    df = pd.read_csv(uploaded_file, sep=';')

    st.write("Data Preview", df.head())

    # ✅ THEN copy
    df_copy = df.copy()

    # Encoding
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()

    for col in df_copy.columns:
        if df_copy[col].dtype == 'object':
            df_copy[col] = le.fit_transform(df_copy[col])

    # Target creation
    if 'G3' in df_copy.columns:
        df_copy['result'] = df_copy['G3'].apply(lambda x: 1 if x >= 10 else 0)
        df_copy = df_copy.drop(['G3'], axis=1)

    X = df_copy.drop('result', axis=1)
    y = df_copy['result']

    # Ensure numeric
    X = X.astype(float)

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier()
    model.fit(X_scaled, y)

    predictions = model.predict(X_scaled)

    df['Result'] = ['Pass' if p == 1 else 'Fail' for p in predictions]

    st.write("Results", df)
