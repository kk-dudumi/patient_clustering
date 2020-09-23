# import all packages
import datetime
import numpy as np
import pandas as pd
import seaborn as sns
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
from sklearn.cluster import KMeans
import sklearn.metrics as sm
from IPython.core.display import display, HTML
from sklearn.metrics.cluster import contingency_matrix
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances
display(HTML("<style>.container { width:100% !important; }</style>"))
from sklearn.manifold import TSNE
from sklearn.metrics import davies_bouldin_score
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
import os
import mysql.connector
connection = mysql.connector.connect(host = '127.0.0.1',user ='root', passwd = 'root', db='patientdata')
print(connection)
df_patient = pd.read_sql_query('''select * from patientdata.measurement ms
join patientdata.medicalhistory mh
on mh.PersonId = ms.PersonId
join patientdata.person pp
on pp.PersonId = ms.PersonId
LIMIT 5000000
''',con=connection)
print(df_patient.head())
print(df_patient.shape)
print(df_patient.columns)
df = df_patient.loc[:,~df_patient.columns.duplicated()]
cat_columns= ['IcdCode','Identifier','Gender','EducationalLevel']
#hot encoding for categorical columns
df_cat = df.loc[:,cat_columns]
df_cat.head()
le = LabelEncoder()
df_encoded = df_cat.apply(le.fit_transform)
df_encoded.head()
#dropping cat columns 
df =df.drop(columns=cat_columns)
df.head()
df_useful = pd.merge(df, df_encoded, left_index=True, right_index=True)
df_useful =df_useful.drop(columns=['CreationDatetime','ModificationDatetime','DateOfBirth', 'Unit','Details'])
df_useful.head()
df_useful= df_useful.dropna(how='any')
df_useful.shape
print(df_useful.nunique())
a = int(input())
plt.figure(figsize=(10, 20))
wcss = []
for i in range(1, 30):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(df_useful)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 30), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()
#Enter number of clusters from elbow method
print("PLEASE ENTER THE NUMBER OF CLUSTERS k FROM THE ELBOW METHOD GRAPH")
k = int(input())

# Fitting K-Means to the dataset
kmeans = KMeans(n_clusters = k, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(df_useful)

#beginning of  the cluster numbering with 1 instead of 0
y_kmeans1=y_kmeans
y_kmeans1=y_kmeans+1
# New Dataframe called cluster
cluster = pd.DataFrame(y_kmeans1)
# Adding cluster to the Dataset1
df_useful['cluster'] = cluster
#Mean of clusters
kmeans_mean_cluster = pd.DataFrame(round(df_useful.groupby('cluster').mean(),1))
print(kmeans_mean_cluster)


#score
labels = kmeans.labels_
print('LABELS - insert random int to continue')
print(labels)
b1 = int(input())
print('davies_bouldin_score')
dbs=davies_bouldin_score(df_useful, labels)
dbs=round(dbs,2)
print(dbs)

print('calinski_harabasz_score')
ch = sm.calinski_harabasz_score(df_useful, labels)
ch=round(ch,2)
print(ch)

# ss=sm.silhouette_score(df_useful, labels, metric='euclidean')
# ss=round(ss,2)
# print(" db score-", dbs, "|", " ch score - ", ch, "|", " ss score - ", ss)




plt.scatter(df_useful.iloc[y_kmeans==0, 0],df_useful.iloc[y_kmeans==0, 1], s=100, c='red', label ='Cluster 1')
plt.scatter(df_useful.iloc[y_kmeans==1, 0], df_useful.iloc[y_kmeans==1, 1], s=100, c='blue', label ='Cluster 2')
plt.scatter(df_useful.iloc[y_kmeans==2, 0], df_useful.iloc[y_kmeans==2, 1], s=100, c='green', label ='Cluster 3')
plt.scatter(df_useful.iloc[y_kmeans==3, 0],df_useful.iloc[y_kmeans==3, 1], s=100, c='orange', label ='Cluster 4')
# plt.scatter(df_useful.iloc[y_kmeans==4, 0], df_useful.iloc[y_kmeans==4, 1], s=100, c='magenta', label ='Cluster 5')
# plt.scatter(df_useful.iloc[y_kmeans==5, 0], df_useful.iloc[y_kmeans==5, 1], s=100, c='pink', label ='Cluster 6')
# plt.scatter(df_useful.iloc[y_kmeans==6, 0],df_useful.iloc[y_kmeans==6, 1], s=100, c='yellow', label ='Cluster 7')
plt.scatter(df_useful.iloc[y_kmeans==1, 0], df_useful.iloc[y_kmeans==1, 1], s=100, c='blue', label ='Cluster 2')
plt.scatter(df_useful.iloc[y_kmeans==2, 0], df_useful.iloc[y_kmeans==2, 1], s=100, c='green', label ='Cluster 3')
Plot the centroid. This time we're going to use the cluster centres  #attribute that returns here the coordinates of the centroid.
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='yellow', label = 'Centroids')
plt.title('Clusters of patients')
plt.xlabel('Measurement_id')
# plt.ylabel('')
plt.show()

plt.scatter(df_useful.iloc[:, 0], df_useful.iloc[:, 2], c=y_kmeans, s=50, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);

#hierarchical clustering
mergings = linkage(df_useful, method='complete')
figure = plt.figure(figsize=(7.5, 5))
dendrogram(
    mergings,
    truncate_mode='lastp',  # show only the last p merged clusters
    p=20,  # show only the last p merged clusters
    leaf_rotation=90.,
    leaf_font_size=12.,
    show_contracted=True,  # to get a distribution impression in truncated branches
)
plt.title('Hierarchical Clustering Dendrogram (Ward)')
plt.xlabel('sample index or (cluster size)')
plt.ylabel('distance')
plt.show()
for k in range(2,11):
    cluster = AgglomerativeClustering(n_clusters=k, affinity='euclidean', linkage='ward').fit(df_useful)
    labels=cluster.labels_
    dbs=davies_bouldin_score(df_useful, labels)
    dbs=round(dbs,2)
    ch = sm.calinski_harabasz_score(df_useful, labels)
    ch=round(ch,2)
    # ss=sm.silhouette_score(df_useful, labels, metric='euclidean')
    # ss=round(ss,2)
    print("Cluster count-", k, "|", " db score-", dbs, "|", " ch score - ", ch, "|")
    k=str(k)
    df_useful['cluster'+k]=labels
    
#TSNE
print("STEP 5")
e = int(input())
model = TSNE(learning_rate=100)
transformed = model.fit_transform(df_useful)
x_axis = transformed[:, 0]
y_axis = transformed[:, 1]
plt.figure(figsize=(10,10))
plt.scatter(x_axis, y_axis)

plt.show()

#DBSCAN
print("STEP 6")
f = int(input())
df_useful = pd.merge(df, df_encoded, left_index=True, right_index=True)
df_useful =df_useful.drop(columns=['CreationDatetime','ModificationDatetime','DateOfBirth','Unit','Details'])
print(df_useful.head())
dbscan = DBSCAN(eps=0.2)
db = dbscan.fit(df_useful)
labels = db.labels_
print(np.unique(labels))
#score
labels = db.labels_
labels = kmeans.labels_
dbs=davies_bouldin_score(df_useful, labels)
dbs=round(dbs,2)
ch = sm.calinski_harabasz_score(df_useful, labels)
ch=round(ch,2)
# ss=sm.silhouette_score(df_useful, labels, metric='euclidean')
# ss=round(ss,2)
print(" db score-", dbs, "|", " ch score - ", ch, "|")
print('contingency_matrix')
cm = contingency_matrix(df_useful, labels)
print(cm)