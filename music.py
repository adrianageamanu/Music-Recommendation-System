import pandas as pd
import ast
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, RobustScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import streamlit as st
import plotly.express as plt

# Set page
st.set_page_config(
    page_title="Music Recommendation System",
    page_icon="🎵",
    layout="wide"
)

# Cache the data loading and preprocessing
@st.cache_data
def load_and_process_data():
    songs = pd.read_csv('songs.csv', sep='\t')
    artists = pd.read_csv('artists.csv', sep='\t')
    features = pd.read_csv('acoustic_features.csv', sep='\t')
    lyrics = pd.read_csv('lyrics.csv', sep='\t')
        
    def extract_artist_name(artist_str):
        # Handle missing values
        if pd.isna(artist_str):
            return None
        try:
            # Convert string representation of a dictionary to an actual dictionary
            artist_dict = ast.literal_eval(artist_str)
            if isinstance(artist_dict, dict):
                return next(iter(artist_dict.values()))
        except:

            # Return None if string cannot be parsed as a dictionary
            return None

    songs['artist'] = songs['artists'].apply(extract_artist_name)
        
    # Merge data
    songs_artists = songs.merge(artists, left_on='artist', right_on='name', how='left')
    songs_data = songs_artists.merge(features, on='song_id', how='left')
    songs_data = songs_data.merge(lyrics, on='song_id', how='left')
        
    # Keep relevant columns
    cols_to_keep = [
            'song_name', 'artist', 'main_genre', 'genres', 
            'danceability', 'energy', 'acousticness',
            'instrumentalness', 'valence', 'tempo', 'loudness', 'speechiness',
            'lyrics'
        ]
    songs_data = songs_data[cols_to_keep]
        
    # Clean data
    songs_data = songs_data.dropna(subset=['danceability', 'energy', 'acousticness', 'valence', 'lyrics'])
    songs_data = songs_data.reset_index(drop=True)
        
    return songs_data
    

@st.cache_data
def prepare_features(songs_data):
    # Numeric features
    num_features = ['danceability', 'energy', 'acousticness', 'instrumentalness',
                    'valence', 'tempo', 'loudness', 'speechiness']
    
    scaler = MinMaxScaler()
    X_numeric = scaler.fit_transform(songs_data[num_features])
    
    # One-hot encode genres
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    X_genres = ohe.fit_transform(songs_data[['main_genre']])
    
    # Combine audio features and genre features
    X_audio_genre = np.concatenate([X_numeric, X_genres], axis=1)
    
    # Process lyrics with TF-IDF
    tfidf = TfidfVectorizer(stop_words='english', max_features=2000)
    X_lyrics = tfidf.fit_transform(songs_data['lyrics'])
    
    # Reduce dimensionality of lyrics
    svd = TruncatedSVD(n_components=50, random_state=42)
    X_lyrics_svd = svd.fit_transform(X_lyrics)
    
    # Combine all features
    alpha = 0.3 # Fixed after testing more values
    X_features = np.concatenate([X_audio_genre * (1 - alpha), X_lyrics_svd * alpha], axis=1)
    
    return X_features

@st.cache_data
def clustering(songs_data):
    cluster_features = songs_data[[
        'danceability', 'energy', 'valence', 'loudness',
        'speechiness', 'acousticness', 'instrumentalness'
    ]].copy()
    
    # Add interaction features
    cluster_features['energy_valence'] = cluster_features['energy'] * cluster_features['valence']
    cluster_features['dance_energy'] = cluster_features['danceability'] * cluster_features['energy']
    
    # Robust scaling to reduce the impact of outliers
    cluster_scaler = RobustScaler()
    cluster_features_scaled = cluster_scaler.fit_transform(cluster_features)
    
    optimal_k = 5 # Fixed for demo
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(cluster_features_scaled) 
    
    # PCA for visualization
    pca = PCA(n_components=2)
    cluster_features_pca = pca.fit_transform(cluster_features_scaled)
    
    return cluster_features_pca, clusters, optimal_k

def enhanced_mood_detection(row):
    valence = row['valence']
    energy = row['energy']
    danceability = row['danceability']

    # Created rules for mood categorization after testing different values
    if danceability > 0.8 and energy > 0.8 and valence > 0.7:
        return 'party'
    if valence < 0.3:
        return 'sad' if energy < 0.5 else 'angry'
    elif valence < 0.6:
        return 'neutral' if energy < 0.7 else 'energetic'
    else:
        return 'happy/calm' if energy < 0.6 else 'happy/energetic'

def parse_playlist_request(text):
    text = text.lower().strip()
    
    # Created key words for moods and genres
    
    mood_map = {
        'sad': ['sad', 'depressive', 'melancholic', 'blue'],
        'happy': ['happy', 'joy', 'upbeat', 'cheer'],
        'angry': ['angry', 'aggressive', 'rage', 'heavy'],
        'energetic': ['energy', 'pump', 'workout', 'exercise'],
        'calm': ['calm', 'chill', 'relax', 'peace'],
        'party': ['party', 'club', 'dance', 'festival']
    }
    
    genre_map = {
        'rock': ['rock', 'alternative', 'indie'],
        'metal': ['metal', 'heavy', 'thrash', 'nu metal'],
        'hip hop': ['hip hop', 'rap', 'trap'],
        'electronic': ['electronic', 'edm', 'house', 'techno'],
        'pop': ['pop'],
        'latin': ['latin', 'reggaeton']
    }
    
    if text == 'random':
        return None, None, True
    
    mood = None
    genre = None
    
    # Check for party first
    if any(kw in text for kw in mood_map['party']):
        return 'party', None, False
    
    # Check for other moods and genres
    for g, keywords in genre_map.items():
        if any(kw in text for kw in keywords):
            genre = g
            break
    
    for m, keywords in mood_map.items():
        if any(kw in text for kw in keywords):
            mood = m
            break
    
    return mood, genre, False

def generate_playlist(df, mood=None, genre=None, length=20):
    # Generating playlist based on mood and genre

    # Copy of the original dataframe
    filtered = df.copy()
    
    # Filter by mood
    if mood == 'party':
        filtered = df[
            (df['danceability'] > 0.8) & 
            (df['energy'] > 0.8) & 
            (df['valence'] > 0.7)
        ]
        if len(filtered) < length:
            filtered = df[
                (df['danceability'] > 0.7) & 
                (df['energy'] > 0.7)
            ]
    else:
        # Filter by genre
        if genre:
            genre_filter = (
                df['main_genre'].str.contains(genre, case=False, na=False) |
                df['genres'].str.contains(genre, case=False, na=False)
            )
            filtered = filtered[genre_filter]
        
        if mood:
            if mood == 'happy':
                filtered = filtered[filtered['mood_category'].isin(['happy/calm', 'happy/energetic'])]
            else:
                filtered = filtered[filtered['mood_category'] == mood]
    
    if filtered.empty and mood:
        if mood == 'happy':
            filtered = df[df['mood_category'].isin(['happy/calm', 'happy/energetic'])]
        else:
            filtered = df[df['mood_category'] == mood]
    
    # Return empty dataframe if empty 
    if filtered.empty:
        return pd.DataFrame()
    
    # Randomly sample songs from the filtered set, up to the requested length
    return filtered.sample(min(len(filtered), length))

def recommend_similar_songs(song_name, artist_name, songs_df, nn_model, feature_matrix, top_n=5):
    # Recommends similar songs using KNN

    # Create a boolean mask to locate rows in songs_df that match both the song and artist
    # Uses case-insensitive matching (`case=False`) and ignores NaNs
    mask = (songs_df['song_name'].str.contains(song_name, case=False, na=False)) & \
           (songs_df['artist'].str.contains(artist_name, case=False, na=False))
    
    if not mask.any():
        return None, f"Song '{song_name}' by '{artist_name}' not found in dataset"
    
    idx = songs_df[mask].index[0]
    distances, indices = nn_model.kneighbors(feature_matrix[idx].reshape(1, -1), n_neighbors=top_n+1)
    recommendations = songs_df.iloc[indices[0][1:]][['song_name', 'artist']]
    
    return recommendations, None

# Streamlit App
def main():
    st.title("🎵 Music Recommendation System")
    st.markdown("---")
    
    # Load data
    with st.spinner("Loading music data..."):
        songs_data = load_and_process_data()
    
    if songs_data is None:
        st.stop()
    
    # Prepare features
    with st.spinner("Preparing..."):
        X_features = prepare_features(songs_data)
        cluster_features_pca, clusters, optimal_k = clustering(songs_data)
        
        # Add mood categories and clusters
        songs_data['mood_category'] = songs_data.apply(enhanced_mood_detection, axis=1)
        songs_data['cluster'] = clusters
        
        # Train recommendation model
        nn = NearestNeighbors(metric='cosine', algorithm='auto')
        nn.fit(X_features)
    
    # Sidebar for navigation
    st.sidebar.title("Menu")
    page = st.sidebar.selectbox("Choose a feature:", [
        "🎧 Song Recommendations", 
        "🎶 Playlist Generator", 
        "📊 Data Visualization",
        "🔍 Explore Dataset"
    ])
    
    if page == "🎧 Song Recommendations":
        st.header("Song Recommendations")
        st.write("Get similar song recommendations based on a input song")
        
        col1, col2 = st.columns(2)
        
        with col1:
            song_name = st.text_input("Song Name:", placeholder="Enter song name...")
        
        with col2:
            artist_name = st.text_input("Artist Name:", placeholder="Enter artist name...")
        
        top_n = st.slider("Number of recommendations:", 1, 20, 5)
        
        if st.button("Get Recommendations", type="primary"):
            if song_name and artist_name:
                with st.spinner("Finding similar songs..."):
                    recommendations, error = recommend_similar_songs(
                        song_name, artist_name, songs_data, nn, X_features, top_n
                    )
                
                if error:
                    st.error(error)
                else:
                    st.success(f"Found {len(recommendations)} recommendations")
                    
                    # Display song recommendations
                    for i, (_, row) in enumerate(recommendations.iterrows(), 1):
                        st.write(f"**{i}.** {row['song_name']} - *{row['artist']}*")
            else:
                st.warning("Enter both song name and artist name")
    
    elif page == "🎶 Playlist Generator":
        st.header("Playlist Generator")
        st.write("Generate custom playlists based on mood and genre")
        
        # Playlist options
        playlist_type = st.selectbox("Choose playlist type:", [
            "Custom (describe what you want)",
            "Random",
            "Party",
            "Sad",
            "Happy",
            "Energetic",
            "Rock",
            "Pop",
        ])
        
        custom_input = ""
        if playlist_type == "Custom (describe what you want)":
            custom_input = st.text_input("Describe your playlist:", 
                                       placeholder="e.g., 'sad metal', 'happy pop'")
        
        playlist_length = st.slider("Playlist length:", 5, 50, 20)
        
        if st.button("Generate Playlist", type="primary"):
            with st.spinner("Creating your playlist..."):
                if playlist_type == "Custom (describe what you want)":
                    if custom_input:
                        mood, genre, is_random = parse_playlist_request(custom_input)
                    else:
                        st.warning("Describe what kind of playlist you want")
                        st.stop()
                elif playlist_type == "Random":
                    mood, genre, is_random = None, None, True
                else:
                    mood, genre, is_random = parse_playlist_request(playlist_type.lower())
                
                if is_random:
                    playlist = songs_data.sample(playlist_length)
                else:
                    playlist = generate_playlist(songs_data, mood, genre, playlist_length)
                
                if playlist.empty:
                    st.error("Can't generate playlist. Try different options")
                else:
                    st.success(f"Generated playlist with {len(playlist)} songs")
                    
                    # Display playlist
                    st.markdown("### Your Playlist")
                    for i, (_, row) in enumerate(playlist.iterrows(), 1):
                        st.write(f"**{i}.** {row['song_name']} - *{row['artist']}*")
    
    elif page == "📊 Data Visualization":
        st.header("Data Visualization")
        st.write("Explore the music dataset through interactive visualizations")
        
        # Data Visualization menu
        choice = st.selectbox("Choose visualization:", [
            "Cluster Analysis",
            "Audio Features Distribution", 
            "Feature Correlations",
            "Mood Analysis",
            "Genre Distribution"
        ])
        
        if choice == "Cluster Analysis":
            st.subheader("Song Clusters (PCA Projection)")
            
            # Visualize clusters
            fig = plt.scatter(
                x=cluster_features_pca[:, 0],
                y=cluster_features_pca[:, 1],
                color=clusters,
                title="Song Clusters",
                labels={'x': 'Principal Component 1', 'y': 'Principal Component 2'},
                color_continuous_scale='viridis'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.write(f"**Total clusters:** {optimal_k}")

            # Calculates silhouette score to analyze clusers' quality
            st.write(f"**Silhouette Score:** {silhouette_score(cluster_features_pca, clusters):.3f}")
        
        elif choice == "Audio Features Distribution":
            st.subheader("Audio Features Distribution")
            
            feature = st.selectbox("Choose feature:", [
                'danceability', 'energy', 'valence', 'acousticness',
                'instrumentalness', 'speechiness', 'loudness', 'tempo'
            ])
            
            fig = plt.histogram(songs_data, x=feature, nbins=50,
                             title=f"Distribution of {feature.title()}")
            st.plotly_chart(fig, use_container_width=True)

        elif choice == "Feature Correlations":
            st.subheader("How Audio Features Relate to Each Other")
            
            # Correlation matrix of key features
            feature_cols = ['danceability', 'energy', 'valence', 'acousticness', 
                           'instrumentalness', 'speechiness', 'loudness']
            corr_matrix = songs_data[feature_cols].corr()
            
            fig = plt.imshow(
                corr_matrix, 
                labels=dict(color="Correlation"),
                title="Feature Correlation Matrix",
                color_continuous_scale='RdBu'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        elif choice == "Mood Analysis":
            st.subheader("Mood Categories")
            
            mood_counts = songs_data['mood_category'].value_counts()
            
            fig = plt.pie(values=mood_counts.values, names=mood_counts.index,
                        title="Distribution of Mood Categories")
            st.plotly_chart(fig, use_container_width=True)
        
        elif choice == "Genre Distribution":
            st.subheader("Genre Distribution")
            
            top_genres = songs_data['main_genre'].value_counts().head(10)
            
            fig = plt.bar(x=top_genres.values, y=top_genres.index,
                        orientation='h', title="Top 10 Genres")
            st.plotly_chart(fig, use_container_width=True)
    
    elif page == "🔍 Explore Dataset":
        st.header("Dataset Explorer")
        st.write("Search through the music dataset!")
        
        # Search bar
        search_term = st.text_input("Search songs or artists:", placeholder="Enter...")
        
        if search_term:
            mask = (songs_data['song_name'].str.contains(search_term, case=False, na=False)) | \
                   (songs_data['artist'].str.contains(search_term, case=False, na=False))
            filtered_songs = songs_data[mask]
            st.write(f"Found {len(filtered_songs)} matching songs")
        else:
            filtered_songs = songs_data
        
        # Filter by mood and genre
        col1, col2 = st.columns(2)
        
        with col1:
            selected_mood = st.selectbox("Filter by mood:", 
                                       ['All'] + list(songs_data['mood_category'].unique()))
        
        with col2:
            selected_genre = st.selectbox("Filter by genre:",
                                        ['All'] + list(songs_data['main_genre'].dropna().unique()[:20]))
        
        
        # Apply filters
        if selected_mood != 'All':
            filtered_songs = filtered_songs[filtered_songs['mood_category'] == selected_mood]
        
        if selected_genre != 'All':
            filtered_songs = filtered_songs[filtered_songs['main_genre'] == selected_genre]
        
        
        # Display results
        st.write(f"Showing {len(filtered_songs)} songs")
        
        if not filtered_songs.empty:
            display_cols = ['song_name', 'artist', 'main_genre', 'mood_category', 
                          'energy', 'valence', 'danceability']
            st.dataframe(filtered_songs[display_cols], use_container_width=True)
    
    # Footer
    st.markdown("---")

if __name__ == "__main__":
    main()