import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("employee.csv")

df_high = df[df["Salary"] > 55000]

mean_age_high = df_high["Age"].mean()
mean_age = df["Age"].mean()
print(f"Average age of salary > 55000 is {mean_age_high:.1f}.")
print(f"Average age of all is {mean_age:.1f}.")

df.plot(x="Age", y="Salary", kind="scatter")
plt.show()



