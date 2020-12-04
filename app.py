from flask import Flask
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.cluster import KMeans
import requests
import json

app = Flask(__name__)

@app.route("/")
def home():
    api_url = 'http://localhost:9191'
    r = requests.get(api_url + '/beneficiarios/direcciones')
    r.raise_for_status()
    beneficiarios = r.json()
    #json_data = json.loads(beneficiarios)
    #print(beneficiarios)
    
    df = pd.DataFrame(beneficiarios)
    df = df.astype({'dirLatGps':'float64', 'dirLonGps':'float64'})
    #print(df.dtypes)
    #plt.scatter(df['dirLatGps'],df['dirLonGps'])
    #plt.xlim(-12.21, -11.83)
    #plt.ylim(-77.16, -76.93)
    #plt.show
    
    x = df.iloc[:,3:5]
    #print(x)

    kmeans = KMeans(4)
    kmeans.fit(x)
    identified_clusters = kmeans.fit_predict(x)
    #print(identified_clusters)

    data_with_clusters = df.copy()
    data_with_clusters['cluster'] = identified_clusters
    data_with_clusters_2 = data_with_clusters[['id', 'dirLatGps', 'dirLonGps', 'cluster']]    
    print(data_with_clusters_2)

    return data_with_clusters_2.to_json(orient='records')