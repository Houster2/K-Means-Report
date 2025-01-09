import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

danh_sach_bai_hat = pd.read_csv('data.csv')
danh_sach_the_loai = pd.read_csv('data_by_genres.csv')
danh_sach_theo_nam = pd.read_csv('data_by_year.csv')

ten_dac_trung = ['acousticness', 'danceability', 'energy', 'instrumentalness', 
'liveness', 'loudness', 'speechiness', 'tempo', 'valence', 'duration_ms', 'explicit', 'key', 'mode', 'year']

X = danh_sach_bai_hat[ten_dac_trung]
y = danh_sach_bai_hat['popularity']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=10)
danh_sach_bai_hat['cluster'] = kmeans.fit_predict(X_scaled)

pca = PCA(n_components=2)
pca_result = pca.fit_transform(X_scaled)
danh_sach_bai_hat['pca1'] = pca_result[:, 0]
danh_sach_bai_hat['pca2'] = pca_result[:, 1]

fig = px.scatter(danh_sach_bai_hat, x='pca1', y='pca2', color='cluster', hover_data=['name', 'cluster'])
fig.show()

auth_manager = SpotifyClientCredentials(client_id='your-client-id', client_secret='your-client-secret')
sp = spotipy.Spotify(auth_manager=auth_manager)

def tim_bai_hat(ten, nam):
    ket_qua = sp.search(q=f'track: {ten} year: {nam}', limit=1)
    if ket_qua['tracks']['items']:
        return ket_qua['tracks']['items'][0]
    return None

def de_xuat_bai_hat(ten_bai_hat, nam_bai_hat, so_luong_bai_hat=5):
    bai_hat = tim_bai_hat(ten_bai_hat, nam_bai_hat)
    if bai_hat:
        dac_trung_bai_hat = np.array([bai_hat['popularity'], bai_hat['duration_ms']])
        khoang_cach = cdist([dac_trung_bai_hat], X_scaled, 'cosine')
        de_xuat = np.argsort(khoang_cach[0])[:so_luong_bai_hat]
        return danh_sach_bai_hat.iloc[de_xuat][['name', 'year', 'cluster']]
    return []

bai_hat_de_xuat = de_xuat_bai_hat('Em Gái Mưa', 2018)
print(bai_hat_de_xuat)
