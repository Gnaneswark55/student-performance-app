# Preprocessing
df_copy = df.copy()

# Convert categorical to numeric
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

for col in df_copy.columns:
    if df_copy[col].dtype == 'object':
        df_copy[col] = le.fit_transform(df_copy[col])

# Create target
if 'G3' in df_copy.columns:
    df_copy['result'] = df_copy['G3'].apply(lambda x: 1 if x >= 10 else 0)
    df_copy = df_copy.drop(['G3'], axis=1)

# Split features and target
X = df_copy.drop('result', axis=1)
y = df_copy['result']

# 🔥 IMPORTANT: ensure all numeric
X = X.astype(float)

# Scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
