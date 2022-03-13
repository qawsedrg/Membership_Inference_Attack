import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="darkgrid")

data = pd.read_csv("../NLP/sub_rnn_imdb", header=None)

sns.relplot(x=0, y=1, kind="line", data=data)
plt.show()
