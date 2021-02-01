import spotipy
import pprint
import spotipy.util as util
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from kneed import KneeLocator
from sklearn.cluster import KMeans
from flask import Flask, request, jsonify, render_template


app = Flask(__name__)

@app.route('/', methods = ['GET', 'POST'])
def home():
    if request.method == 'GET':
        return render_template('index.html')
    else:
        username = request.form['username']
        playlist_uri = request.form['uri']
        client_id = request.form['client_id']
        client_secret = request.form['client_secret']
        redirect_uri = 'http://localhost:8090/callback'
        #return render_template('index.html', message = 'Got You Input..' )
        token, sp = generate_token(username, client_id, client_secret, redirect_uri)
        #return render_template('index.html', message = 'Accessing your playlist..' )
        playlist_name, names, artists, uris = get_playlist_info(username, playlist_uri, sp)
        #return render_template('index.html', message = 'Now I know the details..' )
        df = pd.DataFrame(columns=['name', 'artist', 'track_URI', 'acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'loudness', 'speechiness', 'tempo', 'valence', 'playlist'])
        df = get_features_for_playlist(df, username, playlist_uri, sp)
        non_features = ['name', 'artist', 'track_URI', 'playlist']
        track_info = df[non_features]
        df_X = df.drop(columns=non_features)
        X_std  = scaler(df_X)
        n_comps, scores_pca = dim_reduction_pca(X_std)
        n_clusters = optimal_k_for_kmeans(scores_pca)
        kmeans_pca = Kmeans_enabler(n_clusters, scores_pca)
        df['Cluster'] = kmeans_pca.labels_
        df_seg_pca_kmeans = pd.concat([df_X.reset_index(drop=True), pd.DataFrame(scores_pca)], axis=1)
        df_seg_pca_kmeans.columns.values[(-1*n_comps):] = ["Component " + str(i+1) for i in range(n_comps)]
        df_seg_pca_kmeans['Cluster'] = kmeans_pca.labels_
        plot_clusters(df_seg_pca_kmeans)
        df['Cluster'] = df_seg_pca_kmeans['Cluster']
        df_simplifed = df[['name', 'artist', 'Cluster']]
        display =  df_simplifed.set_index('Cluster').T.to_dict('list')
        
        return render_template('index.html', message = display)



def generate_token(username, client_id, client_secret, redirect_uri):
    token = util.prompt_for_user_token(username, scope='playlist-read-private', client_id=client_id, client_secret=client_secret, redirect_uri=redirect_uri)
    sp = spotipy.Spotify(auth=token)
    return token, sp

# A function to extract track names and URIs from a playlist
def get_playlist_info(username, playlist_uri, sp):
    # initialize vars
    offset = 0
    tracks, uris, names, artists = [], [], [], []

    # get playlist id and name from URI
    playlist_id = playlist_uri.split(':')[2]
    playlist_name = sp.user_playlist(username, playlist_id)['name']

    # get all tracks in given playlist (max limit is 100 at a time --> use offset)
    while True:
        results = sp.user_playlist_tracks(username, playlist_id, offset=offset)
        tracks += results['items']
        if results['next'] is not None:
            offset += 100
        else:
            break
        
    # get track metadata
    for track in tracks:
        names.append(track['track']['name'])
        artists.append(track['track']['artists'][0]['name'])
        uris.append(track['track']['uri'])
    
    return playlist_name, names, artists, uris

# Extract features from each track in a playlist
def get_features_for_playlist(df, username, uri, sp):
  
    # get all track metadata from given playlist
    playlist_name, names, artists, uris = get_playlist_info(username, uri, sp)
    
    # iterate through each track to get audio features and save data into dataframe
    for name, artist, track_uri in zip(names, artists, uris):
        
        # access audio features for given track URI via spotipy 
        audio_features = sp.audio_features(track_uri)

        # get relevant audio features
        feature_subset = [audio_features[0][col] for col in df.columns if col not in ["name", "artist", "track_URI", "playlist"]]

        # compose a row of the dataframe by flattening the list of audio features
        row = [name, artist, track_uri, *feature_subset, playlist_name]
        df.loc[len(df.index)] = row
    return df

def scaler(df_X):
    scaler = StandardScaler()
    X_std = scaler.fit_transform(df_X)
    return X_std

def dim_reduction_pca(X_std):
    pca = PCA()
    pca.fit(X_std)
    evr = pca.explained_variance_ratio_
    for i, exp_var in enumerate(evr.cumsum()):
        if exp_var >= 0.8:
            n_comps = i + 1
            break
    pca = PCA(n_components=n_comps)
    pca.fit(X_std)
    scores_pca = pca.transform(X_std)
    return n_comps, scores_pca

def optimal_k_for_kmeans(scores_pca):
    wcss = []
    max_clusters = 21
    for i in range(1, max_clusters):
        kmeans_pca = KMeans(i, init='k-means++', random_state=42)
        kmeans_pca.fit(scores_pca)
        wcss.append(kmeans_pca.inertia_)
      
    # programmatically locate the elbow
    n_clusters = KneeLocator([i for i in range(1, max_clusters)], wcss, curve='convex', direction='decreasing').knee
    return n_clusters

def Kmeans_enabler(n_clusters, scores_pca):
    kmeans_pca = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
    kmeans_pca.fit(scores_pca)
    return kmeans_pca

def plot_clusters(df_seg_pca_kmeans):
    x = df_seg_pca_kmeans['Component 2']
    y = df_seg_pca_kmeans['Component 1']
    fig = plt.figure(figsize=(10, 8))
    sns.scatterplot(x, y, hue=df_seg_pca_kmeans['Cluster'], palette = 'Set2')
    plt.title('Clusters by PCA Components', fontsize=20)
    plt.xlabel("Component 2", fontsize=18)
    plt.ylabel("Component 1", fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    fig.savefig('./static/img/clusters-v.png')







if __name__ =='__main__':
    app.run(debug=True)