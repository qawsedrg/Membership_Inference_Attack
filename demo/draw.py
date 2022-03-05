import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="darkgrid")

data = pd.read_csv("swap_tran", header=None)

sns.relplot(x=0, y=1, kind="line", data=data)
plt.show()
