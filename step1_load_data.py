import pandas as pd

df = pd.read_csv("vocal_gender_features_new.csv")

print("Shape:", df.shape)
print("Columns:", df.columns.tolist()[:5], "...")
print("Missing values:", df.isnull().sum().sum())
print("Sample:")
print(df.head(2))
