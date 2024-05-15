from flask import Flask, jsonify, request, render_template
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import hstack
from flask_cors import CORS

from bandits_ucb import ContextualBandit
from bandits_ucb import get_trail_features
from bandits_ucb import get_user_features
from rectrek import get_preds_tfidf

# Load pre-computed data structures from your `rectrek.py` file
data = None  # Placeholder for loading data
scaler = None  # Placeholder for loading scaler
tfidf_vectorizer = None  # Placeholder for loading TF-IDF vectorizer
count_vectorizer = None  # Placeholder for loading Count vectorizer
combined_matrix = None  # Placeholder for loading combined matrix
trek_data = None
user_features_df = None
user_ratings_df = None
read_dict = None

# Load pre-computed data structures in a separate function
def load_data_structures():
    global data, scaler, tfidf_vectorizer, count_vectorizer, combined_matrix,trek_data,user_features_df,user_ratings_df,read_dict
    # Load data from CSV file (replace path if needed)
    data = pd.read_csv('/home/ashish/Desktop/checks_frontend/PathFinder/backend/data/Treks_data.csv')
    trek_data = pd.read_csv('/home/ashish/Desktop/checks_frontend/PathFinder/backend/data/Treks_data.csv')
    user_features_df = pd.read_csv('/home/ashish/Desktop/checks_frontend/PathFinder/backend/data/user_features.csv')
    user_ratings_df = pd.read_csv('/home/ashish/Desktop/checks_frontend/PathFinder/backend/data/user_trek_ratings.csv')
    read_dict = np.load('/home/ashish/Desktop/checks_frontend/PathFinder/backend/data/dict.npy',allow_pickle='TRUE').item()
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

# def get_trail_features(data=trek_data):
#     trail_features = data[['Trail_name', 'Difficulty', 'Length', 'Tags', 'Est_time']]

#     difficulty_one_hot = pd.get_dummies(trail_features['Difficulty'], prefix='')
#     trail_features = pd.concat([trail_features, difficulty_one_hot], axis=1)
#     trail_features.drop('Difficulty', axis=1, inplace=True)

#     bins = [0.2, 3.9, 7.2, 13.7, 341.7]  # Adjust bin ranges as needed
#     trail_features['Length'] = pd.cut(trail_features['Length'], bins=bins, labels=['len_0', 'len_1', 'len_2', 'len_3'])
#     est_length_one_hot = pd.get_dummies(trail_features['Length'])
#     trail_features = pd.concat([trail_features, est_length_one_hot], axis=1)
#     trail_features.drop('Length', axis=1, inplace=True)

#     bins = [26.1, 88.0, 159.0, 294.0, 6655.28]  # Adjust bin ranges as needed
#     trail_features['Est_time'] = pd.cut(trail_features['Est_time'], bins=bins, labels=['time_0', 'time_1', 'time_2', 'time_3'])
#     est_time_one_hot = pd.get_dummies(trail_features['Est_time'])
#     trail_features = pd.concat([trail_features, est_time_one_hot], axis=1)
#     trail_features.drop('Est_time', axis=1, inplace=True)

#     trail_features = trail_features.rename(columns={'_Easy': 'easy', '_Hard': 'hard', '_Moderate': 'medium'})

#     # Convert boolean values to integers (0s and 1s)
#     columns_to_convert = ['easy', 'hard', 'medium', 'time_0', 'time_1', 'time_2', 'time_3', 'len_0', 'len_1', 'len_2', 'len_3']
#     trail_features[columns_to_convert] = trail_features[columns_to_convert].astype(int)
#     trail_features = trail_features[['Trail_name', 'Tags', 'len_0', 'len_1', 'len_2', 'len_3', 'time_0', 'time_1', 'time_2', 'time_3', 'easy', 'medium', 'hard']]

#     tags_tfidf = tfidf_vectorizer.fit_transform(trail_features['Tags'])
#     tags_tfidf_df = pd.DataFrame(tags_tfidf.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
#     trail_features_with_tfidf = pd.concat([trail_features, tags_tfidf_df], axis=1)
#     trail_features_with_tfidf.drop('Tags', axis=1, inplace=True)

#     return trail_features_with_tfidf


def get_preds_cucb(tags,difficulty,est_time,length):
    # Create a ContextualBandit instance for each request
    global bandit,features,action
    bandit = ContextualBandit(get_user_features(tags,difficulty,est_time,length), get_trail_features(), user_ratings_df, epsilon=0.1, alpha=0.5)
    print("came here")
    # Run the bandit algorithm to get action (trail prediction) and features
    action, features = bandit.run_bandit()
    print("came here again")
    # Get the recommended trail name from trail_features based on the action (index)
    recommended_trail_name = read_dict[action]
    print(recommended_trail_name)
    return recommended_trail_name


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

@app.route('/rate', methods=['POST'])
def rate():
    # Extract user rating from the form data
    user_data = request.get_json()
    rating = float(user_data.get("rating", 0.0))

    # Retrieve the features used for the recommendation (stored during prediction?)
    recommended_trail_features = features # Replace with how you store features

    # Update the bandit model with rating and features
    bandit.update(action, rating, recommended_trail_features)
    return jsonify({'message': 'Rating submitted successfully!'})

# Endpoint for recommendations using TF-IDF
@app.route("/recommend_cucb", methods=["POST"])
def recommend_cucb():
    # Get user input in JSON format
    user_data = request.get_json()
    tags = user_data.get("tags", "")  # Handle potential missing keys
    difficulty = user_data.get("difficulty", "")
    est_time = float(user_data.get("est_time", 0.0))  # Convert to float explicitly
    length = float(user_data.get("length", 0.0))  # Convert to float explicit
    print(type(length))
    # Call the recommendation function and handle potential errors
    try:
        trail_name= get_preds_cucb(tags, difficulty, est_time, length)
        
        return jsonify({"recommendations": trail_name})
    except Exception as e:
        print(f"Error generating recommendations: {e}")
        return jsonify({"error": "An error occurred while generating recommendations."}), 500


if __name__ == "__main__":
    app.run(debug=True)
