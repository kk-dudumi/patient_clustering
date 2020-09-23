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



#Get all patients with the profile attributes we will use
df_patient = pd.read_sql_query('''select PersonId, DateOfBirth, Gender, EducationalLevel from patientdata.person
''',con=connection)

#Convert DateOfBirth to Age

df_patient['DateOfBirth'] = pd.to_datetime(df_patient.DateOfBirth)

def calculate_age(born):
    born = datetime.strptime(born, "%d.%m.%Y").date()
    today = date.today()
    return today.year - born.year - ((today.month, today.day) < (born.month, born.day))

df_patient['Age'] = df_patient['DateOfBirth'].apply(lambda x: calculate_age(x))
df_patient = df_patient.drop('DateOfBirth', 1)
print(df_patient.head())
#Create IcdCode Column for each patient -initialise with empty list
df_patient['IcdCodes'] =  np.empty((len(df_patient), 0)).tolist()
df_patient.head()
#Populate with ICdCodes list foreach user/patient

def icd_by_person(pid):
    tmp_list = pd.read_sql_query('''select IcdCode from patientdata.medicalhistory where patientdata.medicalhistory.PersonId = %s ''',  con=connection, params = [pid])
    return list(tmp_list.values.flatten())

df_patient['IcdCodes'] = df_patient['PersonId'].apply(lambda x: icd_by_person(x))
print(df_patient.head())