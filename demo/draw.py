import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="darkgrid")

data = pd.read_csv("../data/iris_epsilon.csv")
data1 = data[['epsilon', 'accuracy1']]
data1.rename(columns={'accuracy1': 'accuracy'}, inplace=True)
data1['type'] = "Shadowmodel"
data2 = data[['epsilon', 'accuracy2']]
data2.rename(columns={'accuracy2': 'accuracy'}, inplace=True)
data2['type'] = "Targetmodel"
data3 = pd.concat([data1, data2], axis=0)

sns.relplot(x="epsilon", y="accuracy", kind="line", data=data3, hue="type")
plt.show()
