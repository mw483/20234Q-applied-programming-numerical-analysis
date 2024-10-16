from sklearn import datasets
diabetes = datasets.load_diabetes()
print(diabetes["feature_names"])
print(diabetes["target"])

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

dia = datasets.load_diabetes()
X = pd.DataFrame(dia.data, columns=dia.feature_names)
y = pd.DataFrame(dia.target, columns=["target"])
df = pd.concat([X, y], axis=1)

print(df.head())

corr = df.corr()
fig = plt.figure()
sns.heatmap(corr, vmin=-1, vmax=1, annot=True)
plt.show()

from sklearn import datasets, linear_model
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

x = df[["bmi"]]

model = linear_model.LinearRegression()
model.fit(x, y)
y_pred = model.predict(x)

r2 = r2_score(y, y_pred)

print(f"R2: {r2:.3f}")

# plot
plt.figure(figsize=(5, 5))
plt.scatter(df["bmi"], df["target"])
plt.plot(x_test, model.predict(x_test), color="red")
plt.xlabel("bmi")
plt.ylabel("target")
plt.title("bmi vs target (Linear Regression)")
plt.show()