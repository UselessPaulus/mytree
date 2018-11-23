import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
from sklearn import preprocessing

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.preprocessing import Imputer

plt.figure(figsize=(12, 12))
X = pd.read_csv('X_test.csv', engine='python', sep=';', index_col=False).as_matrix()
#Нормализация данных
X = preprocessing.scale(X)
print(X)

y_pred = KMeans(n_clusters=6).fit_predict(X)

plt.subplot(221)
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.title("Incorrect Number of Blobs")
plt.show()