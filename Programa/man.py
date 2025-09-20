
from flask import Flask, session, redirect, url_for, request
import os
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from spotipy.cache_handler import FlaskSessionCacheHandler

app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(64)

client_id = "f36b47d37468476caad14fc8d8ceb074"  
client_secret = ""  ##NECESARIO ENCRIPTAR EL SECRETO POR SEGURIDAD
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
def home():
    if not sp_oauth.validate_token(cache_handler.get_cached_token()):
        auth_url = sp_oauth.get_authorize_url()
        return redirect(auth_url)
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Inicio app</title>
        <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
                background-color: #121212;
                color: #ffffff;
                margin: 0;
                display: flex;
                flex-direction: column;
                justify-content: center;
                align-items: center;
                min-height: 100vh;
                padding: 20px;
                box-sizing: border-box;
            }}
            .container {{
                background-color: #1e1e1e;
                padding: 40px;
                border-radius: 12px;
                box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5);
                text-align: center;
                max-width: 600px;
                width: 100%;
            }}
            h1 {{
                color: #1DB954;
                margin-bottom: 30px;
                font-size: 2.5em;
            }}
            .button-group {{
                display: flex;
                flex-direction: column;
                gap: 20px;
            }}
            .boton-recientes {{
                background-color: #1DB954;
                color: #ffffff;
                padding: 15px 30px;
                border: none;
                border-radius: 50px;
                text-decoration: none;
                font-weight: bold;
                font-size: 1.2em;
                transition: background-color 0.3s, transform 0.2s;
                box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3);
            }}
            .boton-recientes:hover {{
                background-color: #1ED760;
                transform: translateY(-2px);
            }}
            .boton-artistas {{
                background-color: #535353;
                color: #ffffff;
                padding: 15px 30px;
                border: none;
                border-radius: 50px;
                text-decoration: none;
                font-weight: bold;
                font-size: 1.2em;
                transition: background-color 0.3s, transform 0.2s;
                box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3);
            }}
            .boton-artistas:hover {{
                background-color: #727272;
                transform: translateY(-2px);
            }}
            @media (max-width: 768px) {{
                .container {{
                    padding: 20px;
                }}
                h1 {{
                    font-size: 2em;
                }}
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Bienvenido</h1>
            <div class="button-group">
                <a href="{}" class="boton-recientes">Canciones escuchadas recientemente</a>
                <a href="{}" class="boton-artistas">Top artistas</a>
            </div>
        </div>
    </body>
    </html>
    """.format(url_for('get_recently_played'), url_for('get_top_artists'))

@app.route('/callback')
def callback():
    sp_oauth.get_access_token(request.args['code'])
    return redirect(url_for('home'))

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('home'))

@app.route('/get_recently_played')
def get_recently_played():
    if not sp_oauth.validate_token(cache_handler.get_cached_token()):
        auth_url = sp_oauth.get_authorize_url()
        return redirect(auth_url)
    recently_played = sp.current_user_recently_played()
    recently_played_list = []
    for rp in recently_played['items']:
        track_name = rp['track']['name']
        artist_id = rp['track']['artists'][0]['id']
        artist_name = rp['track']['artists'][0]['name']
        artist_info = sp.artist(artist_id)
        genres = artist_info.get('genres', [])
        genre = genres[0] if genres else 'Undefined'
        recently_played_list.append({
            'name': track_name,
            'artist': artist_name,
            'genre': genre
        })
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Escuchadas recientemente</title>
        <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
                background-color: #121212;
                color: #ffffff;
                margin: 0;
                display: flex;
                justify-content: center;
                align-items: center;
                min-height: 100vh;
                padding: 20px;
                box-sizing: border-box;
            }}
            .container {{
                background-color: #1e1e1e;
                padding: 40px;
                border-radius: 12px;
                box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5);
                text-align: center;
                max-width: 600px;
                width: 100%;
            }}
            h1 {{
                color: #1DB954;
                margin-bottom: 20px;
                font-size: 2.5em;
            }}
            ul {{
                list-style-type: none;
                padding: 0;
                margin: 0;
            }}
            li {{
                background-color: #282828;
                border-radius: 8px;
                padding: 15px;
                margin-bottom: 10px;
                display: flex;
                flex-direction: column;
                align-items: flex-start;
                transition: transform 0.2s;
            }}
            li:hover {{
                transform: translateY(-3px);
            }}
            .track-name {{
                font-weight: bold;
                font-size: 1.1em;
                color: #ffffff;
            }}
            .artist-info {{
                font-size: 0.9em;
                color: #b3b3b3;
                margin-top: 5px;
            }}
            .genre {{
                font-style: italic;
                color: #b3b3b3;
            }}
            .logout-button {{
                background-color: #535353;
                color: #ffffff;
                padding: 10px 20px;
                border: none;
                border-radius: 20px;
                text-decoration: none;
                font-weight: bold;
                margin-top: 20px;
                display: inline-block;
                transition: background-color 0.2s;
            }}
            .logout-button:hover {{
                background-color: #727272;
            }}
            @media (max-width: 768px) {{
                .container {{
                    padding: 20px;
                }}
                h1 {{
                    font-size: 2em;
                }}
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Escuchadas recientemente</h1>
            <ul>
                {list_items}
            </ul>
            <a href="/" class="logout-button">Home</a>
        </div>
    </body>
    </html>
    """
    list_items = ""
    for track in recently_played_list:
        list_items += f"""
        <li>
            <span class="track-name">{track['name']}</span>
            <span class="artist-info">by {track['artist']}</span>
            <span class="genre">({track['genre']})</span>
        </li>
        """
    return html_content.format(list_items=list_items)

@app.route('/get_top_artists')
def get_top_artists():
    if not sp_oauth.validate_token(cache_handler.get_cached_token()):
        auth_url = sp_oauth.get_authorize_url()
        return redirect(auth_url)
    top_artists = sp.current_user_top_artists(limit=10, time_range='short_term')
    top_artists_list = []
    for artist in top_artists['items']:
        artist_name = artist['name']
        genres = artist.get('genres', [])
        genre = genres[0] if genres else 'Undefined'
        
        top_artists_list.append({
            'name': artist_name,
            'genre': genre
        })

    # HTML template for Top artistas with basic styling
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Top artistas</title>
        <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
                background-color: #121212;
                color: #ffffff;
                margin: 0;
                display: flex;
                justify-content: center;
                align-items: center;
                min-height: 100vh;
                padding: 20px;
                box-sizing: border-box;
            }}
            .container {{
                background-color: #1e1e1e;
                padding: 40px;
                border-radius: 12px;
                box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5);
                text-align: center;
                max-width: 600px;
                width: 100%;
            }}
            h1 {{
                color: #1DB954;
                margin-bottom: 20px;
                font-size: 2.5em;
            }}
            ul {{
                list-style-type: none;
                padding: 0;
                margin: 0;
            }}
            li {{
                background-color: #282828;
                border-radius: 8px;
                padding: 15px;
                margin-bottom: 10px;
                display: flex;
                flex-direction: column;
                align-items: flex-start;
                transition: transform 0.2s;
            }}
            li:hover {{
                transform: translateY(-3px);
            }}
            .artist-name {{
                font-weight: bold;
                font-size: 1.1em;
                color: #ffffff;
            }}
            .genre {{
                font-style: italic;
                font-size: 0.9em;
                color: #b3b3b3;
                margin-top: 5px;
            }}
            .logout-button {{
                background-color: #535353;
                color: #ffffff;
                padding: 10px 20px;
                border: none;
                border-radius: 20px;
                text-decoration: none;
                font-weight: bold;
                margin-top: 20px;
                display: inline-block;
                transition: background-color 0.2s;
            }}
            .logout-button:hover {{
                background-color: #727272;
            }}
            @media (max-width: 768px) {{
                .container {{
                    padding: 20px;
                }}
                h1 {{
                    font-size: 2em;
                }}
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Top 10 artistas</h1>
            <ul>
                {list_items}
            </ul>
            <a href="/" class="logout-button">Home</a>
        </div>
    </body>
    </html>
    """
    
    # Generate list items from the fetched data
    list_items = ""
    for artist in top_artists_list:
        list_items += f"""
        <li>
            <span class="artist-name">{artist['name']}</span>
            <span class="genre">({artist['genre']})</span>
        </li>
        """

    return html_content.format(list_items=list_items)

# 7. Run the Application
#    - The app will run on http://127.0.0.1:5000/
if __name__ == "__main__":
    app.run(debug=True)

