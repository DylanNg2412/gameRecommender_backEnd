from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics. pairwise import linear_kernel
from scipy.sparse import hstack, csr_matrix
import os
import re

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

# Compute sparse feature matrices to avoid building a huge dense similarity table at startup
tfidf_tag = compute_tfidf('tags')
tfidf_genre = compute_tfidf('genres')

# Combine all matrices as sparse data
matrix_combined = hstack((tfidf_tag, tfidf_genre)).tocsr()
additional_features = csr_matrix(df[['peak_ccu', 'user_reviews_total']].fillna(0).to_numpy())
matrix_combined_with_features = hstack((matrix_combined, additional_features)).tocsr()

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
def parse_price(val):
    # handle missing values
    if pd.isna(val):
        return 0.0
    s = str(val).strip()
    if not s:
        return 0.0
    # common free markers
    if s.lower() in ('free', 'free to play'):
        return 0.0
    # remove currency prefix/symbols and commas
    s = s.replace('RM', '').replace(',', '').strip()
    # extract first numeric occurrence
    m = re.search(r"(\d+(?:\.\d+)?)", s)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return 0.0
    return 0.0

df['price_myr'] = df['price_myr'].apply(parse_price)

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

        # Get recommendations by comparing one sparse row against the dataset
        idx = indices[selected_game]
        similarity_scores = linear_kernel(matrix_combined_with_features[idx], matrix_combined_with_features).flatten()
        sim_scores = sorted(list(enumerate(similarity_scores)), key=lambda x: x[1], reverse=True)[1:11]
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