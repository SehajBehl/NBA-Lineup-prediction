import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

def create_position_dataset(data, missing_position):
    """
    Creates a dataset for training/testing with a specific position missing.
    """
    # Copy data to avoid modifying original
    position_data = data.copy()
    
    # Define column groups
    home_players = [f'home_{i}' for i in range(5)]
    away_players = [f'away_{i}' for i in range(5)]
    model_features = ['season', 'home_team', 'away_team', 'starting_min']
    
    # Remove the missing position from input features
    input_home_players = [p for p in home_players if p != missing_position]
    input_features = model_features + input_home_players + away_players
    target_column = missing_position
    
    # Store original team names
    original_home_teams = position_data['home_team'].copy()
    original_away_teams = position_data['away_team'].copy()
    game_ids = position_data['game'].copy()
    
    # Select features and target
    X = position_data[input_features].copy()
    y = position_data[target_column].copy()
    
    return X, y, input_features, target_column, game_ids, original_home_teams, original_away_teams

def train_position_model(X, y, input_features, target_column, game_ids, original_home_teams, original_away_teams):
    """
    Trains a model for a specific missing position.
    """
    # Store original data
    X_orig = X.copy()
    
    # Encode categorical variables
    categorical_columns = ['home_team', 'away_team'] + [col for col in X.columns if 'home_' in col or 'away_' in col]
    label_encoders = {}
    
    for col in categorical_columns:
        label_encoders[col] = LabelEncoder()
        X[col] = label_encoders[col].fit_transform(X[col])
    
    # Encode target
    label_encoders[target_column] = LabelEncoder()
    y = label_encoders[target_column].fit_transform(y)
    
    # Split the data with reference data
    X_train, X_test, y_train, y_test, game_train, game_test, home_team_train, home_team_test, away_team_train, away_team_test = train_test_split(
        X, y, game_ids, original_home_teams, original_away_teams, test_size=0.2, random_state=42
    )
    
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
    results['game'] = game_test
    results['season'] = X_test['season']
    results['home_team'] = home_team_test  # Use original team names
    results['away_team'] = away_team_test  # Use original team names
    results['actual_player'] = label_encoders[target_column].inverse_transform(y_test)
    results['predicted_player'] = label_encoders[target_column].inverse_transform(rf_predictions)
    
    # Calculate feature importances
    feature_importances = pd.DataFrame(
        rf_model.feature_importances_,
        index=input_features,
        columns=['importance']
    ).sort_values('importance', ascending=False)
    
    accuracy = accuracy_score(y_test, rf_predictions)
    
    return rf_model, label_encoders, scaler, results, feature_importances, accuracy

def save_results(results, importances, position):
    """
    Saves results to Excel, overwriting existing files
    """
    filename = f'nba_predictions_results_{position}.xlsx'
    try:
        # Use mode='w' to overwrite existing file
        with pd.ExcelWriter(filename, mode='w') as writer:
            results.to_excel(writer, sheet_name='Predictions', index=False)
            importances.to_excel(writer, sheet_name='Feature Importance')
        print(f"Results saved successfully to {filename}")
        return True
    except Exception as e:
        print(f"Error saving results for {position}: {str(e)}")
        return False

# Main execution
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

# Filter for winning home team samples
combined_data = combined_data[combined_data['outcome'] == 1]

print(f"\nNumber of winning home team samples: {len(combined_data)}")
print(f"Seasons covered: {combined_data['season'].unique().tolist()}")

# Train models for each possible missing position
models = {}
results_all = {}
for position in [f'home_{i}' for i in range(5)]:
    print(f"\nTraining model for missing position: {position}")
    
    X, y, input_features, target_column, game_ids, original_home_teams, original_away_teams = create_position_dataset(combined_data, position)
    model, encoders, scaler, results, importances, accuracy = train_position_model(
        X, y, input_features, target_column, game_ids, original_home_teams, original_away_teams
    )
    
    models[position] = {
        'model': model,
        'encoders': encoders,
        'scaler': scaler,
        'feature_importances': importances,
        'accuracy': accuracy
    }
    results_all[position] = results
    
    print(f"Model Accuracy for {position}: {accuracy:.4f}")
    
    # Save results (will overwrite existing files)
    save_results(results, importances, position)

print("\nAll models trained and results saved successfully!")