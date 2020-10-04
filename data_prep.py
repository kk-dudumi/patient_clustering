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
# print(df_patient['DateOfBirth'])


def calculate_age(born):
    born = datetime.datetime.strptime(str(born), "%Y-%m-%d %H:%M:%S").date()
    today = datetime.date.today()
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
#OneHotEncoding for the IcdCodes
df_icd_code_patient= pd.read_sql_query('''select distinct  IcdCode, PersonId from patientdata.medicalhistory''', con=connection ) 
df_icd_code_patient.head()
df_icd_code_patient.shape
df_icd_multicoded = pd.crosstab(df_icd_code_patient['PersonId'],df_icd_code_patient['IcdCode']).rename_axis(None,axis=1)
df_icd_multicoded.head()
df_patient_icd=pd.merge(df_patient,df_icd_multicoded, on ="PersonId")
df_patient_icd.head()

#Get distinct Measurement ids - SNOMED clinical terms are used
df_snomed_ids = pd.read_sql_query('''select distinct SnomedId from patientdata.measurement
''',con=connection)
print(df_snomed_ids)

for col in df_snomed_ids.values.flatten():
    df_patient[col] = 0
#Get median value by measurement type for each patient
def measurement_median_by_person(df_patient):
    for pid in df_patient['PersonId'].values.flatten():
        print(pid)
        pid = str(pid)
        for mid in df_snomed_ids.values.flatten():
            tmp_meas_list = pd.read_sql_query('''select Value from patientdata.measurement where patientdata.measurement.PersonId = %s AND patientdata.measurement.SnomedId = %s''',  con=connection, params = [pid, mid])
            df_patient.loc[0,'{}'.format(mid)] = tmp_meas_list.mean(axis = 0).get('Value')
    return df_patient

measurement_median_by_person(df_patient)
print(df_patient.head())
#Save pre-processed dataset to pkl to be given as input to all clustering algorithms
df_patient.to_pickle("./dataset.pkl")