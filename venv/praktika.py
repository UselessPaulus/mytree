import pandas as pd
import numpy as np
from scipy import stats
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
#df = pd.read_csv('C:\\cvs2.csv')
df = pd.read_csv('C:\\cvs3.csv')
print(df.head(5819))
df[(np.abs(stats.zscore(df)) < 5820).all(axis=1)]
print(df)



# Иерархическая кластеризация
mergings = linkage(df, method='complete')

#Дендрограмма
dendrogram(mergings,
           #labels=varieties,
           leaf_rotation=90,
           leaf_font_size=6,
           )

plt.show()