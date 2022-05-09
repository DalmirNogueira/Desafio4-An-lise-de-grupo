import numpy as np
import pandas as pd
import os
from sklearn.cluster import DBSCAN
from sklearn import metrics

df = pd.read_csv ('Alterado.csv')

db = DBSCAN(eps=.4, min_samples=4).fit(df)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)
#Silhouette
print(metrics.silhouette_score(df, labels, metric='euclidean'))

df2 = pd.read_csv ('df2.csv')
print(metrics.f1_score(df2, labels, average='weighted'))

