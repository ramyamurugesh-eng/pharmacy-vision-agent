import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

df = pd.read_csv("metadata.csv")

print(df.shape)
print(df.columns.tolist())
print(df.head(10))
print()
print("--- Class_Label unique count and sample ---")
print(df["Class_Label"].nunique())
print(df["Class_Label"].unique()[:20])
print()
print("--- Image_Type values ---")
print(df["Image_Type"].unique())
print()
print("--- Environment values ---")
print(df["Environment"].unique())
print()
print("--- Lighting values ---")
print(df["Lighting"].unique())