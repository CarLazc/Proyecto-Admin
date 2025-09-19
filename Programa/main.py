import spotipy
from spotipy.oauth2 import SpotifyOAuth
from spotipy.cache_handler import FlaskSessionCacheHandler
from flask import Flask, session, redirect, url_for, request
from collections import Counter
import random
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
def authorization():
    if not sp_oauth.validate_token(cache_handler.get_cached_token()):
        auth_url = sp_oauth.get_authorize_url()
        return redirect(auth_url)
    return redirect(url_for('get_recently_played'))

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('home'))

@app.route('/callback')
def callback():
    sp_oauth.get_access_token(request.args['code'])
    return redirect(url_for('get_recently_played'))

def get_genre_recommendation(genre_counts, all_genres_from_artists):
    """
    Recomienda un género que aparezca en los artistas pero represente menos del 10% del total
    """
    total_tracks = sum(genre_counts.values())
    threshold = total_tracks * 0.1  # 10% del total
    
    # Géneros que aparecen menos del 10%
    underrepresented_genres = []
    for genre in all_genres_from_artists:
        if genre_counts.get(genre, 0) < threshold:
            underrepresented_genres.append(genre)
    
    if underrepresented_genres:
        # Remover duplicados y seleccionar uno al azar
        unique_underrepresented = list(set(underrepresented_genres))
        return random.choice(unique_underrepresented)
    else:
        return "No hay géneros subrepresentados disponibles"

@app.route('/get_recently_played')  # Corregí el guión
def get_recently_played():
    if not sp_oauth.validate_token(cache_handler.get_cached_token()):
        auth_url = sp_oauth.get_authorize_url()
        return redirect(auth_url) 
    
    recently_played = sp.current_user_recently_played() 
    recently_played_tracks = []
    all_genres = []  # Para almacenar todos los géneros encontrados
    all_genres_from_artists = []  # Todos los géneros de los artistas (para recomendación)
    
    for rp in recently_played['items']:
        track_name = rp['track']['name']
        artist = rp['track']['artists'][0]
        artist_name = artist['name']
        
        # Obtener información del artista
        artist_info = sp.artist(artist['id'])
        genres = artist_info.get('genres', [])
        
        # Tomar el primer género o 'Indefinido'
        primary_genre = genres[0] if genres else 'Indefinido'
        
        # Guardar todos los géneros del artista para recomendación
        all_genres_from_artists.extend(genres)
        
        # Agregar el género principal a la lista
        all_genres.append(primary_genre)
        
        recently_played_tracks.append((track_name, artist_name, primary_genre))
    
    # Contar géneros
    genre_counts = Counter(all_genres)
    
    # Obtener recomendación de género
    recommended_genre = get_genre_recommendation(genre_counts, all_genres_from_artists)
    
    # Crear estadísticas de géneros
    total_tracks = len(recently_played_tracks)
    genre_stats = []
    for genre, count in genre_counts.most_common():
        percentage = (count / total_tracks) * 100
        genre_stats.append(f"{genre}: {count} canciones ({percentage:.1f}%)")
    
    # Crear HTML de respuesta
    recently_played_html = '<h2>Canciones Reproducidas Recientemente:</h2><br>'
    recently_played_html += '<br>'.join(f'{name}: {artist} ({genre})' for name, artist, genre in recently_played_tracks)
    
    recently_played_html += '<br><br><h2>Estadísticas de Géneros:</h2><br>'
    recently_played_html += '<br>'.join(genre_stats)
    
    recently_played_html += f'<br><br><h2>Género Recomendado:</h2><br>'
    recently_played_html += f'<strong>{recommended_genre}</strong>'
    recently_played_html += '<br><small>(Basado en géneros de tus artistas que representan menos del 10% de tu música reciente)</small>'
    
    return recently_played_html 

if __name__ == "__main__":
    app.run(debug=True)