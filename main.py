import spotipy
from spotipy.oauth2 import SpotifyOAuth
from spotipy.cache_handler import FlaskSessionCacheHandler
from flask import Flask, session, redirect, url_for, request
import os

app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(64)

client_id = "f36b47d37468476caad14fc8d8ceb074"
client_secret = "165835c2e70e4195bae2eadd00c65fed"
redirect_uri = "http://127.0.0.1:5000/callback"
scope = "user-read-recently-played"

cache_handler = FlaskSessionCacheHandler(session)
sp_oauth = SpotifyOAuth(client_id=client_id,
                        client_secret=client_secret,
                        redirect_uri=redirect_uri,
                        scope=scope,
                        cache_handler=cache_handler,
                        show_dialog=True)
sp = spotipy.Spotify(auth_manager=sp_oauth)

@app.route('/')
def home():
    if not sp_oauth.validate_token(cache_handler.get_cached_token()):
        auth_url = sp_oauth.get_authorize_url()
        return redirect(auth_url)
    return redirect(url_for('get_recently_played'))

@app.route('/callback')
def callback():
    sp_oauth.get_access_token(request.args['code'])
    return redirect(url_for('get_recently_played'))

@app.route('/get_recently-played')
def get_recently_played():
    if not sp_oauth.validate_token(cache_handler.get_cached_token()):
        auth_url = sp_oauth.get_authorize_url()
        return redirect(auth_url) 
    recently_played = sp.current_user_recently_played() 
    recently_played_tracks = []
    for rp in recently_played['items']:
        track_name = rp['track']['name']
        artist = rp['track']['artists'][0]
        artist_name = artist['name']
        artist_info = sp.artist(artist['id'])
        genres = artist_info.get('genres', [])
        genre = genres[0] if genres else 'Indefinido'
        recently_played_tracks.append((track_name, artist_name, genre))
    recently_played_html = '<br>'.join(f'{name}: {artist} ({genre})' for name, artist, genre in recently_played_tracks)
    return recently_played_html 

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('home'))

if __name__ == "__main__":
    app.run(debug=True)