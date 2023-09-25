#!/usr/bin/env python
# coding: utf-8

# # Klasifikasi Pekerjaan Berdasarkan Deskripsi Pekerjaan dari Postingan JobStreet

# In[22]:


import os, types
import pandas as pd
from botocore.client import Config
import ibm_boto3

def __iter__(self): return 0

# @hidden_cell
# The following code accesses a file in your IBM Cloud Object Storage. It includes your credentials.
# You might want to remove those credentials before you share the notebook.
cos_client = ibm_boto3.client(service_name='s3',
    ibm_api_key_id='452SULhML4jgWMfDx5A8fGfohBRinuUqAnVAnloGA7XJ',
    ibm_auth_endpoint="https://iam.cloud.ibm.com/oidc/token",
    config=Config(signature_version='oauth'),
    endpoint_url='https://s3.private.us-south.cloud-object-storage.appdomain.cloud')

bucket = 'mellinassandbox-donotdelete-pr-o0spccfohrwkp2'
object_key = 'capstone_jobstreet.csv'

body = cos_client.get_object(Bucket=bucket,Key=object_key)['Body']
# add missing __iter__ method, so pandas accepts body as file-like object
if not hasattr(body, "__iter__"): body.__iter__ = types.MethodType( __iter__, body )

df = pd.read_csv(body, low_memory=False, on_bad_lines='skip', nrows=50000)
df


# In[25]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import time

# Mulai pengukuran waktu running
start_time = time.time()

# Menampilkan jumlah total baris dalam dataset
total_rows = len(df)
print("Total baris dalam dataset:", total_rows)

# Pilih kolom yang dibutuhkan
selected_columns = ["description", "jobTitle"]
df_cleaned = df[selected_columns]

# Menghapus baris yang memiliki data "description" atau "jobTitle" yang kosong (NaN)
df_cleaned = df_cleaned.dropna()

# Menggabungkan data yang memiliki kombinasi yang sama dari "description" dan "jobTitle" menjadi satu kelas
df_cleaned = df_cleaned.groupby(['description', 'jobTitle']).first().reset_index()

# Proses preprocessing teks
tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
X = tfidf_vectorizer.fit_transform(df_cleaned['description'])
y = df_cleaned['jobTitle']  # Job title sebagai label target

# Membagi dataset menjadi data pelatihan (training) dan data pengujian (testing) dengan perbandingan 70% dan 30%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Membangun model Decision Tree dengan pengaturan parameter
decision_tree = DecisionTreeClassifier(max_depth=3, min_samples_split=6, min_samples_leaf=3)
decision_tree.fit(X_train, y_train)

# Membuat prediksi dan mengevaluasi model 
y_pred = decision_tree.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Visualisasi pohon keputusan dengan label deskripsi pada simpul akar
plt.figure(figsize=(15, 10))
plot_tree(
    decision_tree,
    filled=True,
    feature_names=tfidf_vectorizer.get_feature_names_out(),
    class_names=y.unique(),
    rounded=True,
    fontsize=9,
    impurity=True,
    proportion=True,
    precision=1,
    node_ids=True,
    label="all",
)
plt.show()

# Menghentikan pengukuran waktu dan mencetak waktu runtime
end_time = time.time()
runtime = end_time - start_time
print("Runtime:", runtime, "detik")


# In[ ]:




