import pandas as pd

df = pd.read_csv("vocal_gender_features_new.csv")
print("Label dtype:", df['label'].dtype)
print("Unique labels:", df['label'].unique())
print("All numeric:", df.drop(columns='label').dtypes.unique())

