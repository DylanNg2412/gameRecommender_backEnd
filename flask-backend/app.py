from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics. pairwise import linear_kernel
import os

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, 'data', 'recommendation_data.csv')

df = pd.read_csv(CSV_PATH)

# Helper function for normalization
def normalize(series):
    return (series - series.min()) / (series.max() - series.min())

# Helper function for TF-IDF computation
def compute_tfidf(column):
    tfid = TfidfVectorizer(stop_words='english')
    return tfid.fit_transform(df[column].fillna(''))

# Normalize additional features
df['peak_ccu'] = normalize(df['peak_ccu'])
df['user_reviews_total'] = normalize(df['user_reviews_total'])

# Compute TF-IDF matrices
dense_matrix_tag = compute_tfidf('tags').toarray()
dense_matrix_genre = compute_tfidf('genres').toarray()

# Combine all matrices
matrix_combined = np.hstack((dense_matrix_tag, dense_matrix_genre))
additional_features = df[['peak_ccu', 'user_reviews_total']].fillna(0).to_numpy()
matrix_combined_with_features = np.hstack((matrix_combined, additional_features))

# Compute cosine similarity
cosine_sim = linear_kernel(matrix_combined_with_features, matrix_combined_with_features)

# Create an index for game names
indices = pd.Series(df. index, index=df['name']. str.lower()).drop_duplicates()

# Generalized label mapping function
def map_label(value, thresholds, labels):
    for threshold, label in zip(thresholds, labels):
        if value <= threshold:
            return label
    return labels[-1]  # Default to the last label

# Define thresholds and labels for popularity and reviews
popularity_thresholds = [0.2, 0.5]
popularity_labels = ['Low Popularity', 'Moderately Popular', 'Highly Popular']

reviews_thresholds = [0.1, 0.3]
reviews_labels = ['Few Reviews', 'Some Reviews', 'Many Reviews']

# Map popularity and reviews to user-friendly labels
df['popularity'] = df['peak_ccu'].apply(lambda x: map_label(x, popularity_thresholds, popularity_labels))
df['reviews'] = df['user_reviews_total'].apply(lambda x: map_label(x, reviews_thresholds, reviews_labels))

# Ensure price is numeric and apply user-friendly labels
df['price_myr'] = pd.to_numeric(df['price_myr'], errors='coerce').fillna(0)

def price_label(price):
    return 'Free' if price == 0 else f"RM{price:.2f}"

df['price'] = df['price_myr'].apply(price_label)

# Flask route for recommendations
@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        data = request.json
        selected_game = data. get("selectedGame", "").strip().lower()
        
        if not selected_game or selected_game not in indices:
            return jsonify({"error": "Game not found"}), 404

        # Get recommendations
        idx = indices[selected_game]
        sim_scores = sorted(list(enumerate(cosine_sim[idx])), key=lambda x: x[1], reverse=True)[1:11]
        game_indices = [i[0] for i in sim_scores]
        max_score = max(sim_scores, key=lambda x: x[1])[1]

        recommendations = df[['name', 'release_date', 'price', 'tags', 'genres', 'popularity', 'reviews']].iloc[game_indices]
        recommendations['score'] = [(score / max_score * 100).round(2) for _, score in sim_scores]

        return jsonify({"recommendations": recommendations.to_dict(orient="records")}), 200

    except Exception as e: 
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

# Updated endpoint - only returns game names for autocomplete
@app.route('/games', methods=['GET'])
def get_games():
    try:
        # Only return the game names, sorted alphabetically
        games_list = sorted(df['name'].unique().tolist())
        return jsonify({"games": games_list}), 200
    except Exception as e:
        return jsonify({"error": f"Failed to fetch games: {str(e)}"}), 500

# New endpoint - search games by query (optional, for better performance)
@app.route('/games/search', methods=['GET'])
def search_games():
    try:
        query = request.args.get('q', '').lower()
        if not query:
            return jsonify({"games": []}), 200
        
        # Filter games that contain the query string
        filtered_games = df[df['name'].str.lower().str.contains(query, na=False)]['name'].unique().tolist()
        return jsonify({"games": filtered_games[: 50]}), 200  # Limit to 50 results
    except Exception as e: 
        return jsonify({"error":  f"Failed to search games: {str(e)}"}), 500

if __name__ == '__main__': 
    app.run(debug=True)