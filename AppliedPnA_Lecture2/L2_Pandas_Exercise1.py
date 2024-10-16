import pandas as pd

df = pd.read_csv("sales_data.csv")
print(df.sum())
print(df.min())
print(df.max())
print(df.mean())
