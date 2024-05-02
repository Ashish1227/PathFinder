import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import hstack

data = pd.read_csv('/home/ashish/Desktop/checks_frontend/PathFinder/backend/data/Treks_data.csv')
features = ['Tags', 'Difficulty', 'Average_rating', 'Length']

scaler = MinMaxScaler()
data[['Average_rating', 'Length']] = scaler.fit_transform(data[['Average_rating', 'Length']])
data['combined_features'] = data.apply(lambda x: ' '.join([str(x[feat]) for feat in features]), axis=1)

tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2))
tfidf_matrix = tfidf_vectorizer.fit_transform(data['combined_features'])

count_vectorizer = CountVectorizer(ngram_range=(1, 2))
count_matrix = count_vectorizer.fit_transform(data['combined_features'])

def recommend_trails_countv(tags, difficulty, average_rating, length, num_recommendations=5):
    # Combine user input into a feature vector
    user_features = ' '.join([tags, str(difficulty), str(average_rating), str(length)])
    # Transform user input using Count Vectorizer
    user_count = count_vectorizer.transform([user_features])
    # Calculate cosine similarity between user input and all trails
    similarities = cosine_similarity(user_count, count_matrix)
    # Get indices of trails with highest similarity
    top_indices = similarities.argsort()[0][-num_recommendations:][::-1]
    # Get recommended trails
    recommended_trails = data.iloc[top_indices]
    scores = [(data.iloc[i]['Trail_name'], similarities[0][i]) for i in top_indices]
    return recommended_trails[['Trail_name', 'link_AllTrails', 'image', 'Average_rating', 'Difficulty', 'Length', 'Tags']], scores

def recommend_trails_tfidfv(tags, difficulty, average_rating, length, num_recommendations=5):
    # Combine user input into a feature vector
    user_features = ' '.join([tags, difficulty, str(average_rating), str(length)])
    # Transform user input using TF-IDF Vectorizer
    user_tfidf = tfidf_vectorizer.transform([user_features])   
    # Calculate cosine similarity between user input and all trails
    similarities = cosine_similarity(user_tfidf, tfidf_matrix)   
    # Get indices of trails with highest similarity
    top_indices = similarities.argsort()[0][-num_recommendations:][::-1]  
    # Get recommended trails
    recommended_trails = data.iloc[top_indices]  
    scores = [(data.iloc[i]['Trail_name'], similarities[0][i]) for i in top_indices] 
    return recommended_trails[['Trail_name', 'link_AllTrails', 'image', 'Average_rating', 'Difficulty', 'Length', 'Tags']], scores

def get_preds_tfidf(tags,difficulty,average_rating,length):
    recommendations_tfidf, tfidf_scores = recommend_trails_tfidfv(tags, difficulty, average_rating, length)
    recommendations_tfidf[['Average_rating', 'Length']] = scaler.inverse_transform(recommendations_tfidf[['Average_rating', 'Length']])

    return tfidf_scores

def get_preds_count(tags,difficulty,average_rating,length):
    recommendations_count, count_scores = recommend_trails_countv(tags, difficulty, average_rating, length)
    recommendations_count[['Average_rating', 'Length']] = scaler.inverse_transform(recommendations_count[['Average_rating', 'Length']])

    return count_scores

combined_matrix = hstack([tfidf_matrix, count_matrix])

def recommend_trails_combined(tags, difficulty, average_rating, length, num_recommendations=5):
    # Combine user input into a feature vector
    user_features = ' '.join([tags, difficulty, str(average_rating), str(length)])
    # Transform user input using combined matrix
    user_combined = hstack([tfidf_vectorizer.transform([user_features]), 
                            count_vectorizer.transform([user_features])])
    # Calculate cosine similarity between user input and all trails
    similarities = cosine_similarity(user_combined, combined_matrix)
    # Get indices of trails with highest similarity
    top_indices = similarities.argsort()[0][-num_recommendations:][::-1] 
    # Get recommended trails
    recommended_trails = data.iloc[top_indices]
    # Extract scores for the recommended trails
    scores = [(data.iloc[i]['Trail_name'], similarities[0][i]) for i in top_indices]
    return recommended_trails[['Trail_name', 'link_AllTrails', 'image', 'Average_rating', 'Difficulty', 'Length', 'Tags']], scores

def get_preds_cmb(tags,difficulty,average_rating,length):
    recommendations_combined, scores_combined = recommend_trails_combined(tags, difficulty, average_rating, length)

    return scores_combined

