import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn import datasets
diabetes = datasets.load_diabetes()
print(diabetes["feature_names"])
print(diabetes["target"])


dia = datasets.load_diabetes()
X = pd.DataFrame(dia.data, columns=dia.feature_names)
y = pd.DataFrame(dia.target, columns=["target"])
df = pd.concat([X, y], axis=1)

x = df[["bmi"]]

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

model = linear_model.LinearRegression()
model.fit(X_train, y_train)

r2_train = r2_score(y_train, model.predict(X_train))
r2_test  = r2_score(y_test, model.predict(X_test))

print(f"Training R2: {r2_train:.3f}")
print(f"Test R2: {r2_test:.3f}")

# plot
plt.figure(figsize=(5, 5))
plt.scatter(df["bmi"], df["target"])
plt.plot(X_test, model.predict(X_test), color="red")
plt.xlabel("bmi")
plt.ylabel("target")
plt.title("bmi vs target (Linear Regression)")
plt.show()