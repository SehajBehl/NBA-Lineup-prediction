import os
import pandas as pd

# Define the directory containing the data files
data_dir = 'data'

# Get a list of all CSV files in the directory
all_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.csv')]

# Load all CSV files into a list of DataFrames
data_frames = []
for file in all_files:
    try:
        df = pd.read_csv(file)
        data_frames.append(df)
    except Exception as e:
        print(f"Error loading {file}: {e}")

# Combine all DataFrames into one
combined_data = pd.concat(data_frames, ignore_index=True)

# Handle missing values
combined_data['starting_min'] = combined_data['starting_min'].fillna(combined_data['starting_min'].mean())

player_columns = ['home_0', 'home_1', 'home_2', 'home_3', 'home_4', 
                  'away_0', 'away_1', 'away_2', 'away_3', 'away_4']
for col in player_columns:
    combined_data[col] = combined_data[col].fillna('Unknown')

# Standardize player names (strip spaces and lowercase)
for col in player_columns:
    combined_data[col] = combined_data[col].str.strip().str.lower()

# Ensure consistent data types
combined_data = combined_data.astype({
    'game': str,
    'season': int,
    'home_team': str,
    'away_team': str,
    'starting_min': float
}, errors='ignore')

# Remove duplicate rows
combined_data = combined_data.drop_duplicates()

# Validate seasons
print(f"Unique seasons: {combined_data['season'].unique()}")

# Filter data for training (2007–2014) and testing (2015)
training_data = combined_data[combined_data['season'].between(2007, 2014)].copy()
test_data = combined_data[combined_data['season'] == 2015].copy()

# Normalize 'starting_min' using training mean and std
mean_train = training_data['starting_min'].mean()
std_train = training_data['starting_min'].std()

training_data.loc[:, 'starting_min'] = (training_data['starting_min'] - mean_train) / std_train
test_data.loc[:, 'starting_min'] = (test_data['starting_min'] - mean_train) / std_train  # Avoid data leakage

# Select only allowed features
allowed_features = ['game', 'season', 'home_team', 'away_team', 'starting_min', 
                    'home_0', 'home_1', 'home_2', 'home_3', 'home_4',
                    'away_0', 'away_1', 'away_2', 'away_3', 'away_4']

training_data = training_data[allowed_features]
test_data = test_data[allowed_features]

print("Training data columns:" , training_data.columns)
# Display results
print("Training Data Sample:")
print(training_data.head())
print(f"Training Data Rows: {training_data.shape[0]}")

print("\nTest Data Sample:")
print(test_data.head())
print(f"Test Data Rows: {test_data.shape[0]}")
