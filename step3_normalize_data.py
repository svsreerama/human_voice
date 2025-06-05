import pandas as pd
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("vocal_gender_features_new.csv")
X = df.drop("label", axis=1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Before scaling:", X.iloc[0, :3].tolist())
print("After scaling:", X_scaled[0, :3].tolist())

