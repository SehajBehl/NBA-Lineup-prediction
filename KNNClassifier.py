import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from collections import Counter

data_dir = 'data'
all_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.csv')]
data_frames = []
for file in all_files:
    try:
        df = pd.read_csv(file)
        data_frames.append(df)
    except Exception as e:
        print(f"Error loading {file}: {e}")

combined_data = pd.concat(data_frames, ignore_index=True).dropna()
combined_data.drop_duplicates(inplace=True)

combined_data['game'] = combined_data['game'].astype(str)

allowed_features = ['game', 'season', 'home_team', 'away_team', 'starting_min', 
                    'home_0', 'home_1', 'home_2', 'home_3', 'home_4',
                    'away_0', 'away_1', 'away_2', 'away_3', 'away_4']
combined_data = combined_data[allowed_features + ['outcome']]

categorical_columns = ['home_team', 'away_team', 'home_0', 'home_1', 'home_2', 
                       'home_3', 'home_4', 'away_0', 'away_1', 'away_2', 'away_3', 'away_4']

label_encoders = {}
for col in categorical_columns:
    label_encoders[col] = LabelEncoder()
    combined_data[col] = label_encoders[col].fit_transform(combined_data[col])

combined_data['season'] = combined_data['season'].astype('int32')
combined_data['starting_min'] = combined_data['starting_min'].astype('float32')

X = combined_data.drop(columns=['game', 'home_4', 'outcome'])
y = combined_data['home_4']

# 3. Random 80/20 Train-Test Split (using all seasons 2007-2015)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Normalize 'starting_min' using training data statistics
mean_train = X_train['starting_min'].mean()
std_train = X_train['starting_min'].std()
X_train.loc[:, 'starting_min'] = (X_train['starting_min'] - mean_train) / std_train
X_test.loc[:, 'starting_min'] = (X_test['starting_min'] - mean_train) / std_train

# 5. Custom KNN Algorithm Implementation
def custom_knn_predict(X_train, y_train, X_test, k=5):
    """
    Custom K-Nearest Neighbors predictor.
    For each test instance, compute Euclidean distances to all training samples,
    then use a majority vote among the k nearest neighbors.
    
    Parameters:
        X_train (DataFrame): Training features.
        y_train (Series): Training target (encoded player names).
        X_test (DataFrame): Test features.
        k (int): Number of neighbors.
    
    Returns:
        np.array: Array of predictions (numeric labels).
    """
    predictions = []
    X_train_vals = X_train.values
    y_train_vals = y_train.values
    X_test_vals = X_test.values
    # Loop over each test instance
    for test_point in X_test_vals:
        
        distances = np.sqrt(np.sum((X_train_vals - test_point) ** 2, axis=1))
      
        neighbor_indices = np.argsort(distances)[:k]
        neighbor_labels = y_train_vals[neighbor_indices]
        
        most_common = Counter(neighbor_labels).most_common(1)[0][0]
        predictions.append(most_common)
    return np.array(predictions)


k = 1

# 6. Make Predictions on Test Data using the Custom KNN
custom_predictions = custom_knn_predict(X_train, y_train, X_test, k=k)


decoded_predictions = label_encoders['home_4'].inverse_transform(custom_predictions)

decoded_actual = label_encoders['home_4'].inverse_transform(y_test)

# Prepare a DataFrame to show results
results = X_test.copy()
results['actual_home_4'] = decoded_actual
results['predicted_home_4'] = decoded_predictions

print("\nTest Data with Predictions:")
print(results.head())

# Optionally, compute and print accuracy on the test set
accuracy = np.mean(custom_predictions == y_test.values)
print(f"\nCustom KNN Accuracy: {accuracy:.4f}")


