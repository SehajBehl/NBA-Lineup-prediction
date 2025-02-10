import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Data loading and preprocessing remains the same until feature selection
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

print("\nData Range Check:")
print(f"Seasons covered: {combined_data['season'].unique().tolist()}")

combined_data = combined_data[combined_data['outcome'] == 1]
print(f"\nNumber of winning home team samples: {len(combined_data)}")

# Explicitly define our column groups
home_players = ['home_0', 'home_1', 'home_2', 'home_3']  # Exactly 4 home players
away_players = ['away_0', 'away_1', 'away_2', 'away_3', 'away_4']  # Exactly 5 away players
other_features = ['season', 'home_team', 'away_team', 'starting_min']
target_column = 'home_4'

# Combine all input features
input_features = other_features + home_players + away_players

# Select only the columns we want
X = combined_data[input_features].copy()
y = combined_data[target_column].copy()

print("\nInput feature columns:")
print("Home players:", home_players)
print("Away players:", away_players)
print("Other features:", other_features)
print("\nTotal number of input features:", len(input_features))

# Encode categorical variables
categorical_columns = ['home_team', 'away_team'] + home_players + away_players
label_encoders = {}

for col in categorical_columns:
    label_encoders[col] = LabelEncoder()
    X[col] = label_encoders[col].fit_transform(X[col])

# Encode target
label_encoders[target_column] = LabelEncoder()
y = label_encoders[target_column].fit_transform(y)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale only starting_min
scaler = StandardScaler()
X_train['starting_min'] = scaler.fit_transform(X_train[['starting_min']])
X_test['starting_min'] = scaler.transform(X_test[['starting_min']])

# Initialize and train the Random Forest model
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_split=4,
    min_samples_leaf=1,
    max_features='sqrt',
    class_weight='balanced',
    n_jobs=-1,
    random_state=42
)

rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)

# Create results DataFrame
results = pd.DataFrame()
results['season'] = X_test['season']
results['home_team'] = label_encoders['home_team'].inverse_transform(X_test['home_team'])
results['away_team'] = label_encoders['away_team'].inverse_transform(X_test['away_team'])
results['actual_home_4'] = label_encoders[target_column].inverse_transform(y_test)
results['predicted_home_4'] = label_encoders[target_column].inverse_transform(rf_predictions)

# Add this after the results DataFrame creation and before the print statements
# Enhance results DataFrame with more details
results['confidence'] = rf_model.predict_proba(X_test).max(axis=1)

# Get feature importances for each prediction
feature_importances = pd.DataFrame(
    rf_model.feature_importances_,
    index=input_features,
    columns=['importance']
).sort_values('importance', ascending=False)

# Save to Excel with multiple sheets
with pd.ExcelWriter('nba_predictions_results.xlsx') as writer:
    results.to_excel(writer, sheet_name='Predictions', index=False)
    feature_importances.to_excel(writer, sheet_name='Feature Importance')
    
print("\nResults saved to 'nba_predictions_results.xlsx'")

# Calculate accuracy
accuracy = accuracy_score(y_test, rf_predictions)

print("\nRandom Forest Model Accuracy: {:.4f}".format(accuracy))
print("\nTest Data with Predictions:")
print(results.head().to_string(index=True))

# Correct player count verification
print("\nVerification of player counts:")
print(f"Number of home players in input (should be 4): {len(home_players)}")
print(f"Number of away players in input (should be 5): {len(away_players)}")


# Save model and preprocessing objects
# model_dir = 'models'
# if not os.path.exists(model_dir):
#     os.makedirs(model_dir)

# joblib.dump(rf_model, os.path.join(model_dir, 'rf_model.joblib'))
# joblib.dump(label_encoders, os.path.join(model_dir, 'label_encoders.joblib'))
# joblib.dump(scaler, os.path.join(model_dir, 'scaler.joblib'))