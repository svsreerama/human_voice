import pandas as pd

df = pd.read_csv("vocal_gender_features_new.csv")
df.drop("label", axis=1).to_csv("input_features.csv", index=False)
print("✅ input_features.csv saved")

