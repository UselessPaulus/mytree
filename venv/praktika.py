import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.preprocessing import MinMaxScaler, StandardScaler


df = pd.read_csv('C:\\cvs5.csv', delimiter=',')
print(df)
cols = ['instr' , 'class' , 'nb.repeat',  'attendance','difficulty','Q1','Q2','Q3','Q4','Q5','Q6','Q7','Q8','Q9','Q10','Q11','Q12','Q13','Q14','Q15','Q16','Q17','Q18','Q19','Q20','Q21','Q22','Q23','Q24','Q25','Q26','Q27','Q28']

scaler = StandardScaler()
X = scaler.fit_transform(df.drop(['Q1','Q2', 'Q3','Q4','Q5','Q6','Q7','Q8','Q9','Q10','Q11','Q12','Q13','Q14','Q15','Q16','Q17','Q18','Q19','Q20','Q21','Q22','Q23','Q24','Q25','Q26','Q27','Q28'],1))
Y = scaler.fit_transform(df.drop(['instr' , 'class' , 'nb.repeat',  'attendance','difficulty'],1))
# clustering
n_clusters = 5
km = KMeans(n_clusters=n_clusters)

# fit & predict clusters
df['cluster'] = km.fit_predict(X)

# results - we should have 3 clusters: [0,1,2]
print(df)
# cluster's centroids
print(km.cluster_centers_)



plt.figure()
pd.plotting.radviz(df, 'Q1')
plt.show()
#print(df.head(5819))

#list(df)
#print(list(df))
#A = df

#A.reshape((-1,1))
#kmeans = KMeans(n_clusters=2, random_state=0).fit(A)

#labels_array([], dtype=int32)
#kmeans.predict([-1,1])
#array([0, 1], dtype=int32)
#kmeans.cluster_centers_array([[1., 2.],
#[4., 2.]])
#print(kmeans.labels_)
#print(kmeans.predict)
#print(kmeans.cluster_centers_)

#plt.figure(figsize=(6,6))

#y_pred = KMeans(n_clusters=2, random_state=0).fit(A.reshape(-1, 1))

#plt.subplot(221)
#plt.scatter(A[-1], A[1], c=y_pred)

#plt.show()