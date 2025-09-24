import os
from flask import Flask, session, redirect, url_for, request, jsonify
from flask_cors import CORS
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from spotipy.cache_handler import FlaskSessionCacheHandler

app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(64)
CORS(app, supports_credentials=True)
client_id = "f36b47d37468476caad14fc8d8ceb074"
client_secret = "165835c2e70e4195bae2eadd00c65fed"
FRONTEND_URL = "http://127.0.0.1:5500/intra_frontend.html"
redirect_uri = "http://127.0.0.1:5000/callback"
scope = "user-read-recently-played user-top-read"

cache_handler = FlaskSessionCacheHandler(session)
sp_oauth = SpotifyOAuth(
    client_id=client_id,
    client_secret=client_secret,
    redirect_uri=redirect_uri,
    scope=scope,
    cache_handler=cache_handler,
    show_dialog=True
)
sp = spotipy.Spotify(auth_manager=sp_oauth)
@app.route('/')
def login():
    auth_url = sp_oauth.get_authorize_url()
    return redirect(auth_url)

@app.route('/callback')
def callback():
    sp_oauth.get_access_token(request.args['code'])

    return redirect(FRONTEND_URL)

@app.route('/api/top-artists')
def get_top_artists():
    if not sp_oauth.validate_token(cache_handler.get_cached_token()):
        return jsonify({'error': 'Authorization required'}), 401
    top_artists = sp.current_user_top_artists(limit=3, time_range='long_term')
    artists_data = []
    for i, artist in enumerate(top_artists['items']):
        image_url = artist['images'][0]['url'] if artist['images'] else "https://placehold.co/208x208/7c3aed/ffffff?text=Artist"
        shape = "rounded-full" #if i == 1 else "rounded-3xl"
        artists_data.append({
            'name': artist['name'],
            'imageUrl': image_url,
            'shape': shape,
            'description': f"Genero: {artist['genres'][0] if artist['genres'] else 'Indefinido'}"
        })
    return jsonify(artists_data)

@app.route('/api/top-tracks')
def get_top_tracks():
    if not sp_oauth.validate_token(cache_handler.get_cached_token()):
        return jsonify({'error': 'Authorization required'}), 401
    top_tracks = sp.current_user_top_tracks(limit=10, time_range='long_term')
    tracks_data = []
    for track in top_tracks['items']:
        tracks_data.append({
            'name': track['name'],
            'artist': track['artists'][0]['name']
        })
    return jsonify(tracks_data)

@app.route('/api/recently-played')
def get_recently_played():
    if not sp_oauth.validate_token(cache_handler.get_cached_token()):
        return jsonify({'error': 'Authorization required'}), 401
    recently_played = sp.current_user_recently_played(limit=10)
    played_data = []
    for item in recently_played['items']:
        track = item['track']
        played_data.append({
            'name': track['name'],
            'artist': track['artists'][0]['name']
        })
    return jsonify(played_data) 

if __name__ == "__main__":
    app.run(debug=True, port=5000)

