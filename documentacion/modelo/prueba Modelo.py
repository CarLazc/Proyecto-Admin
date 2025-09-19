import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
import numpy as np
import os
import time
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

load_dotenv()

class SpotifyDataCollectorV2:
    def __init__(self):
        self.client_id = os.getenv('SPOTIFY_CLIENT_ID')
        self.client_secret = os.getenv('SPOTIFY_CLIENT_SECRET')
        
        if not self.client_id or not self.client_secret:
            raise ValueError("Credenciales de Spotify no encontradas en archivo .env")
        
        try:
            auth_manager = SpotifyClientCredentials(
                client_id=self.client_id,
                client_secret=self.client_secret
            )
            self.sp = spotipy.Spotify(auth_manager=auth_manager)
            print("Conectado a Spotify API")
        except Exception as e:
            raise ConnectionError(f"Error conectando a Spotify: {e}")
    
    def test_connection(self):
        try:
            results = self.sp.search(q='test', type='track', limit=1)
            print("Conexion a Spotify funcionando")
            return True
        except Exception as e:
            print(f"Error de conexion: {e}")
            return False
    
    def get_artist_detailed_info(self, artist_id):
        try:
            artist = self.sp.artist(artist_id)
            return {
                'genres': artist.get('genres', []),
                'popularity': artist.get('popularity', 0),
                'followers': artist.get('followers', {}).get('total', 0)
            }
        except Exception as e:
            return {'genres': [], 'popularity': 0, 'followers': 0}
    
    def extract_advanced_features(self, track):
        track_name = track['name'].lower()
        
        rock_keywords = ['rock', 'metal', 'punk', 'grunge', 'alternative']
        pop_keywords = ['pop', 'dance', 'party', 'love', 'heart']
        hip_hop_keywords = ['rap', 'hip', 'hop', 'feat', 'ft']
        electronic_keywords = ['electronic', 'remix', 'mix', 'beat', 'drop']
        
        features = {
            'has_rock_keywords': any(word in track_name for word in rock_keywords),
            'has_pop_keywords': any(word in track_name for word in pop_keywords),
            'has_hip_hop_keywords': any(word in track_name for word in hip_hop_keywords),
            'has_electronic_keywords': any(word in track_name for word in electronic_keywords),
            'track_name_length': len(track['name']),
            'has_feat': 'feat' in track_name or 'ft.' in track_name,
            'has_parentheses': '(' in track['name'],
            'year': int(track['album']['release_date'][:4]) if len(track['album']['release_date']) >= 4 else 2020,
        }
        
        return features
    
    def search_tracks_by_genre(self, genre, limit=50, offset=0):
        search_strategies = [
            f'genre:"{genre}" year:2018-2024',
            f'{genre} year:2020-2024',
            f'{genre} popular',
            f'genre:{genre}'
        ]
        
        for strategy in search_strategies:
            try:
                results = self.sp.search(
                    q=strategy,
                    type='track',
                    limit=limit,
                    offset=offset,
                    market='US'
                )
                
                tracks = results['tracks']['items']
                if tracks:
                    valid_tracks = [
                        track for track in tracks 
                        if (track['id'] and 
                            track['popularity'] > 10 and 
                            track['artists'] and
                            track['album']['release_date'])
                    ]
                    if valid_tracks:
                        return valid_tracks
                
                time.sleep(0.3)
                
            except Exception as e:
                continue
        
        return []
    
    def collect_comprehensive_dataset(self, genres, tracks_per_genre=15):
        print(f"Recolectando dataset...")
        print(f"Generos: {', '.join(genres)}")
        
        all_data = []
        
        for genre_idx, genre in enumerate(genres, 1):
            print(f"[{genre_idx}/{len(genres)}] Procesando: {genre}")
            
            genre_tracks = []
            offset = 0
            max_offset = 1000
            
            while len(genre_tracks) < tracks_per_genre and offset < max_offset:
                tracks = self.search_tracks_by_genre(genre, limit=50, offset=offset)
                
                if not tracks:
                    break
                
                genre_tracks.extend(tracks)
                
                offset += 50
                time.sleep(0.5)
            
            final_tracks = genre_tracks[:tracks_per_genre]
            successful_tracks = 0
            
            for track in final_tracks:
                try:
                    artist_info = self.get_artist_detailed_info(track['artists'][0]['id'])
                    time.sleep(0.3)
                    
                    advanced_features = self.extract_advanced_features(track)
                    
                    track_data = {
                        'track_id': track['id'],
                        'track_name': track['name'],
                        'artist_name': track['artists'][0]['name'],
                        'artist_id': track['artists'][0]['id'],
                        'album_name': track['album']['name'],
                        'search_genre': genre,
                        'popularity': track['popularity'],
                        'explicit': int(track['explicit']),
                        'duration_ms': track['duration_ms'],
                        'artist_popularity': artist_info['popularity'],
                        'artist_followers': artist_info['followers'],
                        'release_year': advanced_features['year'],
                        'track_name_length': advanced_features['track_name_length'],
                        'has_feat': int(advanced_features['has_feat']),
                        'has_parentheses': int(advanced_features['has_parentheses']),
                        'has_rock_keywords': int(advanced_features['has_rock_keywords']),
                        'has_pop_keywords': int(advanced_features['has_pop_keywords']),
                        'has_hip_hop_keywords': int(advanced_features['has_hip_hop_keywords']),
                        'has_electronic_keywords': int(advanced_features['has_electronic_keywords']),
                        'artist_genres': ', '.join(artist_info['genres'][:5]),
                        'num_artist_genres': len(artist_info['genres']),
                        'duration_minutes': track['duration_ms'] / 60000,
                        'is_recent': int(advanced_features['year'] >= 2020),
                        'artist_tier': 'high' if artist_info['popularity'] > 70 else 'medium' if artist_info['popularity'] > 40 else 'low'
                    }
                    
                    all_data.append(track_data)
                    successful_tracks += 1
                    
                except Exception as e:
                    continue
            
            print(f"  {successful_tracks} tracks procesados para {genre}")
        
        print(f"Dataset completo: {len(all_data)} tracks")
        return all_data

class GenreRecommendationSystem:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.tfidf_vectorizer = None
        
        self.numeric_features = [
            'popularity', 'explicit', 'duration_ms', 'artist_popularity',
            'artist_followers', 'release_year', 'track_name_length',
            'has_feat', 'has_parentheses', 'has_rock_keywords',
            'has_pop_keywords', 'has_hip_hop_keywords', 'has_electronic_keywords',
            'num_artist_genres', 'duration_minutes', 'is_recent'
        ]
    
    def prepare_data(self, data):
        print("Preparando datos para el modelo...")
        
        df = pd.DataFrame(data)
        df_clean = df.dropna(subset=self.numeric_features + ['search_genre'])
        
        print(f"Datos limpios: {len(df_clean)} samples")
        print("Distribucion por genero:")
        print(df_clean['search_genre'].value_counts())
        
        return df_clean
    
    def create_features(self, df):
        X_numeric = df[self.numeric_features].values
        
        self.tfidf_vectorizer = TfidfVectorizer(max_features=50, stop_words='english')
        artist_genres_text = df['artist_genres'].fillna('')
        X_text = self.tfidf_vectorizer.fit_transform(artist_genres_text).toarray()
        
        X_combined = np.hstack([X_numeric, X_text])
        
        print(f"Features creadas: {X_combined.shape[1]} features totales")
        
        return X_combined
    
    def train(self, df):
        print("Entrenando modelo...")
        
        X = self.create_features(df)
        y = df['search_genre'].values
        
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced'
        )
        
        self.model.fit(X_train, y_train)
        
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Entrenamiento completado!")
        print(f"Accuracy: {accuracy:.3f}")
        
        print("Reporte de clasificacion:")
        target_names = self.label_encoder.classes_
        print(classification_report(y_test, y_pred, target_names=target_names))
        
        return accuracy
    
    def recommend_genre(self, input_data, top_n=3):
        """
        Funcion principal para recomendar generos basado en datos de entrada
        
        input_data: diccionario con las caracteristicas de la cancion
        Ejemplo:
        {
            'popularity': 75,
            'duration_ms': 210000,
            'artist_popularity': 80,
            'artist_followers': 1000000,
            'release_year': 2023,
            'track_name_length': 15,
            'has_feat': 0,
            'has_parentheses': 1,
            'has_rock_keywords': 1,
            'has_pop_keywords': 0,
            'has_hip_hop_keywords': 0,
            'has_electronic_keywords': 0,
            'num_artist_genres': 3,
            'is_recent': 1,
            'explicit': 0,
            'artist_genres': 'rock, alternative rock, indie rock'
        }
        """
        
        if self.model is None:
            raise ValueError("Modelo no entrenado. Ejecuta train() primero.")
        
        # Crear features con valores por defecto
        processed_features = {}
        
        # Features numericas con valores por defecto
        default_values = {
            'popularity': 50,
            'explicit': 0,
            'duration_ms': 200000,
            'artist_popularity': 50,
            'artist_followers': 100000,
            'release_year': 2022,
            'track_name_length': 10,
            'has_feat': 0,
            'has_parentheses': 0,
            'has_rock_keywords': 0,
            'has_pop_keywords': 0,
            'has_hip_hop_keywords': 0,
            'has_electronic_keywords': 0,
            'num_artist_genres': 2,
            'is_recent': 1
        }
        
        # Calcular duration_minutes automaticamente
        duration_ms = input_data.get('duration_ms', default_values['duration_ms'])
        processed_features['duration_minutes'] = duration_ms / 60000
        
        # Aplicar valores de entrada o defaults
        for feature in self.numeric_features:
            if feature != 'duration_minutes':  # ya calculado arriba
                processed_features[feature] = input_data.get(feature, default_values[feature])
        
        # Preparar vector de features numericas
        numeric_vals = [processed_features[feat] for feat in self.numeric_features]
        
        # Features de texto
        artist_genres = input_data.get('artist_genres', '')
        
        # Crear feature vector completo
        X_numeric = np.array([numeric_vals])
        X_text = self.tfidf_vectorizer.transform([artist_genres]).toarray()
        X_combined = np.hstack([X_numeric, X_text])
        
        # Normalizar y predecir
        X_scaled = self.scaler.transform(X_combined)
        probabilities = self.model.predict_proba(X_scaled)[0]
        
        # Obtener top recomendaciones
        top_indices = np.argsort(probabilities)[-top_n:][::-1]
        
        recommendations = []
        for idx in top_indices:
            genre = self.label_encoder.inverse_transform([idx])[0]
            confidence = probabilities[idx]
            recommendations.append({
                'genre': genre,
                'confidence': confidence,
                'percentage': f"{confidence*100:.1f}%"
            })
        
        return recommendations
    
    def save_model(self, filepath='data/genre_recommendation_model.pkl'):
        """Guarda el modelo entrenado"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'tfidf_vectorizer': self.tfidf_vectorizer,
            'numeric_features': self.numeric_features
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Modelo guardado en: {filepath}")
    
    def load_model(self, filepath='data/genre_recommendation_model.pkl'):
        """Carga un modelo previamente entrenado"""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.label_encoder = model_data['label_encoder']
            self.tfidf_vectorizer = model_data['tfidf_vectorizer']
            self.numeric_features = model_data['numeric_features']
            
            print(f"Modelo cargado desde: {filepath}")
            return True
        except Exception as e:
            print(f"Error cargando modelo: {e}")
            return False

def train_model():
    """Entrena un nuevo modelo"""
    try:
        collector = SpotifyDataCollectorV2()
        
        if not collector.test_connection():
            return None
        
        genres = ['rock', 'pop', 'hip-hop', 'jazz', 'electronic', 'country']
        
        print("Recolectando datos...")
        dataset = collector.collect_comprehensive_dataset(genres, tracks_per_genre=15)
        
        if not dataset:
            print("No se pudo recolectar datos")
            return None
        
        os.makedirs('data', exist_ok=True)
        df_raw = pd.DataFrame(dataset)
        df_raw.to_csv('data/spotify_training_dataset.csv', index=False)
        print("Datos de entrenamiento guardados")
        
        model = GenreRecommendationSystem()
        df_clean = model.prepare_data(dataset)
        accuracy = model.train(df_clean)
        
        model.save_model()
        
        return model
        
    except Exception as e:
        print(f"Error: {e}")
        return None

def recommend_genre_for_song(song_data, model_path='data/genre_recommendation_model.pkl'):
    """
    Funcion simple para obtener recomendacion de genero
    
    song_data: diccionario con caracteristicas de la cancion
    """
    model = GenreRecommendationSystem()
    
    if not model.load_model(model_path):
        print("No se pudo cargar el modelo. Entrenando uno nuevo...")
        model = train_model()
        if model is None:
            return None
    
    recommendations = model.recommend_genre(song_data)
    
    print("RECOMENDACIONES DE GENERO:")
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec['genre']}: {rec['percentage']} confianza")
    
    return recommendations

def main():
    print("SISTEMA DE RECOMENDACION DE GENEROS MUSICALES")
    print("=" * 50)
    
    # Entrenar modelo
    model = train_model()
    
    if model is None:
        return
    
    # Ejemplo de uso del sistema de recomendacion
    print("\nEJEMPLO DE RECOMENDACION:")
    print("-" * 30)
    
    ejemplo_cancion = {
        'popularity': 85,
        'duration_ms': 195000,
        'artist_popularity': 90,
        'artist_followers': 5000000,
        'release_year': 2023,
        'track_name_length': 12,
        'has_rock_keywords': 1,
        'has_pop_keywords': 0,
        'has_hip_hop_keywords': 0,
        'has_electronic_keywords': 0,
        'explicit': 0,
        'num_artist_genres': 4,
        'artist_genres': 'rock, alternative rock, indie rock, hard rock'
    }
    
    print("Datos de entrada:")
    for key, value in ejemplo_cancion.items():
        print(f"  {key}: {value}")
    
    recommendations = model.recommend_genre(ejemplo_cancion)
    print(f"\nRECOMENDACIONES:")
    for i, rec in enumerate(recommendations, 1):
        print(f"  {i}. {rec['genre']}: {rec['percentage']}")
    
    print(f"\nPara usar el sistema con nuevos datos, usa la funcion:")
    print(f"recommend_genre_for_song(tus_datos)")

if __name__ == "__main__":
    main()