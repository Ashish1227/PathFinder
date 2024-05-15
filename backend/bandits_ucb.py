import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import csv
import ast


trek_data = pd.read_csv('/home/ashish/Desktop/checks_frontend/PathFinder/backend/data/Treks_data.csv')
user_features = pd.read_csv('/home/ashish/Desktop/checks_frontend/PathFinder/backend/data/user_features.csv')
user_ratings = pd.read_csv('/home/ashish/Desktop/checks_frontend/PathFinder/backend/data/user_trek_ratings.csv')
tfidf_vectorizer = TfidfVectorizer()


length_pref_data = {
    'min': [0.2, 3.9, 7.2, 13.7],
    'max': [3.7, 7.1, 13.5, 341.7]
}
length_pref_df = pd.DataFrame(length_pref_data)
length_pref_df.index = ['len_0', 'len_1', 'len_2', 'len_3']

time_pref_data = {
    'min': [26.102753, 88.0, 159.0, 294.0],
    'max': [87.0, 157.0, 293.0, 6655.278361]
}
time_pref_df = pd.DataFrame(time_pref_data)
time_pref_df.index = ['time_0', 'time_1', 'time_2', 'time_3']

def get_trail_features():
    trail_features = trek_data[['Trail_name', 'Difficulty', 'Length', 'Tags', 'Est_time']]

    difficulty_one_hot = pd.get_dummies(trail_features['Difficulty'], prefix='')
    trail_features = pd.concat([trail_features, difficulty_one_hot], axis=1)
    trail_features.drop('Difficulty', axis=1, inplace=True)

    bins = [length_pref_df['min'].iloc[0], length_pref_df['min'].iloc[1], length_pref_df['min'].iloc[2], length_pref_df['min'].iloc[3], length_pref_df['max'].iloc[3]]

    trail_features['Length'] = pd.cut(trail_features['Length'], bins=bins, labels=['len_0', 'len_1', 'len_2', 'len_3'])
    est_length_one_hot = pd.get_dummies(trail_features['Length'])
    trail_features = pd.concat([trail_features, est_length_one_hot], axis=1)
    trail_features.drop('Length', axis=1, inplace=True)

    bins = [time_pref_df['min'].iloc[0], time_pref_df['min'].iloc[1], time_pref_df['min'].iloc[2], time_pref_df['min'].iloc[3], time_pref_df['max'].iloc[3]]

    trail_features['Est_time'] = pd.cut(trail_features['Est_time'], bins=bins, labels=['time_0', 'time_1', 'time_2', 'time_3'])
    est_time_one_hot = pd.get_dummies(trail_features['Est_time'])
    trail_features = pd.concat([trail_features, est_time_one_hot], axis=1)
    trail_features.drop('Est_time', axis=1, inplace=True)

    trail_features = trail_features.rename(columns={'_Easy': 'easy', '_Hard': 'hard', '_Moderate': 'medium'})

    # Convert boolean values to integers (0s and 1s)
    columns_to_convert = ['easy', 'hard', 'medium', 'time_0', 'time_1', 'time_2', 'time_3', 'len_0', 'len_1', 'len_2', 'len_3']
    trail_features[columns_to_convert] = trail_features[columns_to_convert].astype(int)
    trail_features = trail_features[['Trail_name', 'Tags', 'len_0', 'len_1', 'len_2', 'len_3', 'time_0', 'time_1', 'time_2', 'time_3', 'easy', 'medium', 'hard']]

    tags_tfidf = tfidf_vectorizer.fit_transform(trail_features['Tags'])
    tags_tfidf_df = pd.DataFrame(tags_tfidf.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
    trail_features_with_tfidf = pd.concat([trail_features, tags_tfidf_df], axis=1)
    trail_features_with_tfidf.drop('Tags', axis=1, inplace=True)

    index_trail_dict = dict(zip(trail_features_with_tfidf.index, trail_features_with_tfidf['Trail_name']))
    trail_index_dict = {v: k for k, v in index_trail_dict.items()}
    trail_features_with_tfidf.drop(columns=['Trail_name'], inplace=True)
    return trail_features_with_tfidf

def get_user_features(tags, difficulty, est_time, length, trail_features=get_trail_features()):
    data = {'Difficulty': [difficulty], 'Length': [length], 'Tags': [tags], 'Est_time': [est_time]}
    user_df = pd.DataFrame(data)

    bins = [length_pref_df['min'].iloc[0], length_pref_df['min'].iloc[1], length_pref_df['min'].iloc[2], length_pref_df['min'].iloc[3], length_pref_df['max'].iloc[3]]

    user_df['Length'] = pd.cut(user_df['Length'], bins=bins, labels=['len_0', 'len_1', 'len_2', 'len_3'])
    est_length_one_hot = pd.get_dummies(user_df['Length'])
    user_df = pd.concat([user_df, est_length_one_hot], axis=1)
    user_df.drop('Length', axis=1, inplace=True)

    bins = [time_pref_df['min'].iloc[0], time_pref_df['min'].iloc[1], time_pref_df['min'].iloc[2], time_pref_df['min'].iloc[3], time_pref_df['max'].iloc[3]]

    user_df['Est_time'] = pd.cut(user_df['Est_time'], bins=bins, labels=['time_0', 'time_1', 'time_2', 'time_3'])
    est_time_one_hot = pd.get_dummies(user_df['Est_time'])
    user_df = pd.concat([user_df, est_time_one_hot], axis=1)
    user_df.drop('Est_time', axis=1, inplace=True)

    user_df['easy'] = 0
    user_df['medium'] = 0
    user_df['hard'] = 0
    user_df.loc[user_df['Difficulty'] == 'easy', 'easy'] = 1
    user_df.loc[user_df['Difficulty'] == 'medium', 'medium'] = 1
    user_df.loc[user_df['Difficulty'] == 'hard', 'hard'] = 1
    user_df.drop('Difficulty', axis=1, inplace=True)

    tags_tfidf = tfidf_vectorizer.fit_transform(user_df['Tags'])
    tags_tfidf_df = pd.DataFrame(tags_tfidf.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
    missing_cols = set(trail_features.columns) - set(tags_tfidf_df.columns)
    missing_cols = set(missing_cols) - set(user_df.columns)
    for col in missing_cols:
        tags_tfidf_df[col] = 0
    user_df_with_tfidf = pd.concat([user_df, tags_tfidf_df], axis=1)
    user_df_with_tfidf.drop('Tags', axis=1, inplace=True)

    columns_after_hard = user_df_with_tfidf.columns[user_df_with_tfidf.columns.get_loc('hard')+1:]
    sorted_columns = sorted(columns_after_hard)

    # Now you can rearrange the DataFrame columns
    new_order = list(user_df_with_tfidf.columns[:user_df_with_tfidf.columns.get_loc('hard')+1]) + sorted_columns
    user_df_with_tfidf = user_df_with_tfidf[new_order]
    return user_df_with_tfidf

# tags = 'Cave, walking, forest'
# difficulty = 'hard'
# est_time = 160
# length = 5

# # Call the function with the inputs
# x = get_user_features(tags, difficulty, est_time, length)
# y = get_trail_features()


class ContextualBandit:
    def __init__(self, user_features, trail_features, user_ratings, epsilon, alpha):
        self.user_features = user_features
        self.trail_features = trail_features
        self.user_ratings = user_ratings
        self.num_user_features = len(user_features.columns)
        self.num_trail_features = len(trail_features.columns)
        self.epsilon = epsilon
        self.alpha = alpha
        self.data_file = '/home/ashish/Desktop/checks_frontend/PathFinder/backend/data/bandit_ucb_data.csv'
        self.weights_file = '/home/ashish/Desktop/checks_frontend/PathFinder/backend/data/bandit_ucb_weights.txt'
        self.data = []
        self.temp_data = []
        
        # Load existing data from CSV file if it exists
        if os.path.isfile(self.data_file) and os.path.getsize(self.data_file) > 0:
            self.temp_data = pd.read_csv(self.data_file)
        else:
            self.temp_data = pd.DataFrame(columns=['Features', 'Reward'])
            self.save_data()  

        if not self.temp_data.empty:
            for index, row in self.temp_data.iterrows():
                features = ast.literal_eval(row['Features'])  # Convert string to list
                self.data.append((features, row['Reward']))

        try:
            self.weights = np.loadtxt(self.weights_file)
            if self.weights.size == 0:
                self.weights = None
        except FileNotFoundError:
            self.weights = None

    def save_data(self):
        if isinstance(self.temp_data, pd.DataFrame):
            self.temp_data.to_csv(self.data_file, index=False)
        else:
            # Create an empty DataFrame with correct column names
            empty_df = pd.DataFrame(columns=['Features', 'Reward'])
            empty_df.to_csv(self.data_file, index=False)



    def choose_action(self, features, t=1e-4):
        # Choose action based on current weights
        if self.weights is None:
            action = np.random.randint(len(features))
        else:
            X = np.array([x for x, _ in self.data])
            n = X.shape[1]
            ucb_scores = []
            for feature in features:
                ucb_score = np.dot(self.weights, feature) + self.alpha * np.sqrt(np.dot(feature.T, np.linalg.inv(np.dot(X.T, X) + t * np.eye(n)) @ feature))
                ucb_scores.append(ucb_score)
            action = np.argmax(ucb_scores)
        return action

    def update_weights(self, alpha=1e-2):
        X = np.array([x for x, _ in self.data])
        R = np.array([r for _, r in self.data])
        X_transpose = np.transpose(X)
        n = 156
        self.weights = np.dot(np.dot(np.linalg.inv(np.dot(X_transpose, X) + alpha * np.eye(n)), X_transpose), R)
        np.savetxt(self.weights_file, self.weights)

    def add_data_point(self, features, reward):
        features_as_list = features.tolist()
        self.data.append((features_as_list, reward))
        with open(self.data_file, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([features_as_list, reward])



    def run_bandit(self):
        # Randomly choose user features for task t
        user_feature = self.user_features.iloc[0]
        # Generate trail features for each trail ai
        trail_features = self.trail_features
        # Concatenate user features with each trail feature
        concatenated_features = [np.concatenate((user_feature.values, trail_feat.values)) for _, trail_feat in trail_features.iterrows()]
        
        # # Choose action
        action = self.choose_action(concatenated_features)
        return action, concatenated_features
        
    def update(self, action, r, concatenated_features):
        # Get reward for chosen action
        reward = r
        # Add new data point
        self.add_data_point(concatenated_features[action], reward)
        # Update weights
        self.update_weights()

# user_features = x
# trail_features = y
# user_ratings = user_ratings
# epsilon = 0.1
# # T = 500
# alpha = 0.5  # You may need to tune this parameter
# bandit = ContextualBandit(user_features, trail_features, user_ratings, epsilon, alpha)
# act , feat = bandit.run_bandit()
# print(act)
# print(type(feat[act]))
# bandit.update(act,3.6, feat)