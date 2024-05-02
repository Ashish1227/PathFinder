from flask import Flask, jsonify, request
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import hstack
from flask_cors import CORS

from rectrek import get_preds_tfidf

# Load pre-computed data structures from your `rectrek.py` file
data = None  # Placeholder for loading data
scaler = None  # Placeholder for loading scaler
tfidf_vectorizer = None  # Placeholder for loading TF-IDF vectorizer
count_vectorizer = None  # Placeholder for loading Count vectorizer
combined_matrix = None  # Placeholder for loading combined matrix

# Load pre-computed data structures in a separate function
def load_data_structures():
    global data, scaler, tfidf_vectorizer, count_vectorizer, combined_matrix
    # Load data from CSV file (replace path if needed)
    data = pd.read_csv('/home/ashish/Desktop/checks_frontend/PathFinder/backend/data/Treks_data.csv')
    # Select features for recommendation
    features = ['Tags', 'Difficulty', 'Average_rating', 'Length']
    # Create a MinMaxScaler for normalization
    scaler = MinMaxScaler()
    data[['Average_rating', 'Length']] = scaler.fit_transform(data[['Average_rating', 'Length']])
    # Create a combined feature column for easier processing
    data['combined_features'] = data.apply(lambda x: ' '.join([str(x[feat]) for feat in features]), axis=1)
    # Create TF-IDF vectorizer
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    tfidf_matrix = tfidf_vectorizer.fit_transform(data['combined_features'])
    # Create Count vectorizer
    count_vectorizer = CountVectorizer(ngram_range=(1, 2))
    count_matrix = count_vectorizer.fit_transform(data['combined_features'])
    # Create a combined sparse matrix for recommendations using both TF-IDF and Count features
    combined_matrix = hstack([tfidf_matrix, count_matrix])


# Ensure data structures are loaded before processing requests
load_data_structures()

app = Flask(__name__)
CORS(app)
# Endpoint for recommendations using TF-IDF
@app.route("/recommend_tfidf", methods=["POST"])
def recommend_tfidf():
    # Get user input in JSON format
    user_data = request.get_json()
    tags = user_data.get("tags", "")  # Handle potential missing keys
    difficulty = user_data.get("difficulty", "")
    average_rating = user_data.get("average_rating", 0.0)  # Default value
    length = user_data.get("length", 0.0)

    # Call the recommendation function and handle potential errors
    try:
        scores = get_preds_tfidf(tags, difficulty, average_rating, length)
        return jsonify({"recommendations": scores})
    except Exception as e:
        print(f"Error generating recommendations: {e}")
        return jsonify({"error": "An error occurred while generating recommendations."}), 500

# Endpoint for recommendations using Count Vectorizer
@app.route("/recommend_count", methods=["POST"])
def recommend_count():
    # Get user input in JSON format (similar to recommend_tfidf)
    user_data = request.get_json()
    # ... (extract user data and call get_preds_count)

# Endpoint for recommendations using Combined Matrix
@app.route("/recommend_combined", methods=["POST"])
def recommend_combined():
    # Get user input in JSON format (similar to recommend_tfidf)
    user_data = request.get_json()
    # ... (extract user data and call get_preds_cmb)

if __name__ == "__main__":
    app.run(debug=True)
