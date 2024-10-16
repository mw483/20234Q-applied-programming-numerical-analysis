import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="darkgrid")

# loading dataset
iris = sns.load_dataset("iris")

# scatter plot
sns.scatterplot(data=iris)
plt.show()

# histogram
sns.histplot(iris.petal_length)
plt.show()

# pair plot -- showing all the combination
sns.pairplot(data=iris)
plt.show()


