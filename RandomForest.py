import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,f1_score
from sklearn.pipeline import Pipeline
from config import config
import warnings
from collections import Counter



class SafeLabelEncoder:
    def __init__(self, unknown_value=-1, warn_on_unknown=True):
        self.unknown_value = unknown_value
        self.warn_on_unknown = warn_on_unknown
        self.classes_ = None
        self.mapping_ = {}
        self.frequency_ = None  # Track frequency of labels
        
    def fit(self, y):
        y = np.array(y, dtype=str)
        # Count frequencies to prioritize common players
        counter = Counter(y)
        self.classes_ = np.array([label for label, _ in counter.most_common()])
        self.mapping_ = {label: idx for idx, label in enumerate(self.classes_)}
        self.frequency_ = counter
        return self
        
    def transform(self, y):
        y = np.array(y, dtype=str)
        result = []
        for label in y:
            if label in self.mapping_:
                result.append(self.mapping_[label])
            else:
                # Assign "New_Player" category instead of -1
                if "New_Player" not in self.mapping_:
                    self.mapping_["New_Player"] = len(self.mapping_)
                result.append(self.mapping_["New_Player"])
        return np.array(result)

        
    def fit_transform(self, y):
        return self.fit(y).transform(y)
    
    def inverse_transform(self, y):
        reverse_mapping = {idx: label for label, idx in self.mapping_.items()}
        return np.array([reverse_mapping.get(val, "UNKNOWN") for val in y])

def create_position_dataset(data, missing_position, player_stats=None, team_stats=None):
    """
    Creates a dataset for training/testing with a specific position missing.
    Incorporates additional player and team statistics if provided.
    """
    # Copy data to avoid modifying the original
    position_data = data.copy()
    
    # Define column groups
    home_players = [f'home_{i}' for i in range(5)]
    away_players = [f'away_{i}' for i in range(5)]
    model_features = ['season', 'home_team', 'away_team', 'starting_min']
    
    # Add derived features
    if 'home_score' in position_data.columns and 'away_score' in position_data.columns:
        position_data['score_diff'] = position_data['home_score'] - position_data['away_score']
        model_features.append('score_diff')
    
    # Create team combination feature
    position_data['matchup'] = position_data['home_team'] + '_vs_' + position_data['away_team']
    model_features.append('matchup')
    
    # Create features for player combinations - but limit them to avoid dimensionality issues
    player_combos = []
    for i, player1 in enumerate(home_players):
        if player1 == missing_position:
            continue
        # Only create combos between adjacent positions to limit features
        for player2 in home_players[i+1:i+2]:
            if player2 == missing_position or i+1 >= len(home_players):
                continue
            combo_name = f'combo_{i}_{i+1}'
            position_data[combo_name] = position_data[player1] + '_' + position_data[player2]
            player_combos.append(combo_name)
    
    # Remove the missing position from input features
    input_home_players = [p for p in home_players if p != missing_position]
    input_features = model_features + input_home_players + away_players + player_combos
    target_column = missing_position
    
    # Store original team names and game ids if available
    original_home_teams = position_data['home_team'].copy()
    original_away_teams = position_data['away_team'].copy()
    if 'game' in position_data.columns:
        game_ids = position_data['game'].copy()
    else:
        game_ids = pd.Series(range(len(position_data)))
    
    # Add team statistics if provided
    if team_stats is not None:
        # Merge team stats for home team
        team_stats_home = team_stats.copy()
        team_stats_home = team_stats_home.rename(columns={col: f'home_{col}' if col not in ['team', 'season'] else col for col in team_stats_home.columns})
        team_stats_home = team_stats_home.rename(columns={'team': 'home_team'})
        
        position_data = pd.merge(
            position_data, 
            team_stats_home,
            on=['home_team', 'season'],
            how='left'
        )
        
        # Add columns to features
        for col in team_stats_home.columns:
            if col not in ['home_team', 'season'] and col not in input_features:
                input_features.append(col)
        
        # Same for away team
        team_stats_away = team_stats.copy()
        team_stats_away = team_stats_away.rename(columns={col: f'away_{col}' if col not in ['team', 'season'] else col for col in team_stats_away.columns})
        team_stats_away = team_stats_away.rename(columns={'team': 'away_team'})
        
        position_data = pd.merge(
            position_data, 
            team_stats_away,
            on=['away_team', 'season'],
            how='left'
        )
        
        # Add columns to features
        for col in team_stats_away.columns:
            if col not in ['away_team', 'season'] and col not in input_features:
                input_features.append(col)
    
    # Select features and target, filling NAs
    X = position_data[input_features].copy().fillna(0)
    y = position_data[target_column].copy()
    
    return X, y, input_features, target_column, game_ids, original_home_teams, original_away_teams


def train_position_model(X, y, input_features, target_column, game_ids, original_home_teams, original_away_teams, test_data):
    """
    Trains a model for a specific missing position with enhanced model selection.
    """
    # Copy original X
    X_orig = X.copy()
    
    # Identify categorical and numerical columns
    categorical_columns = ['home_team', 'away_team', 'matchup']
    categorical_columns.extend([col for col in X.columns if col.startswith('home_') or col.startswith('away_') or col.startswith('combo_')])
    categorical_columns = [col for col in categorical_columns if col in X.columns and X[col].dtype == 'object']
    
    numerical_columns = [col for col in X.columns if col not in categorical_columns]
    
    # ✅ Expand Label Encoding to Include Both Train & Test Labels
    all_players = pd.concat([y, test_data[target_column]], ignore_index=True)  # Combine training and test labels
    all_players = all_players.dropna().astype(str)  # Drop NaN & ensure string format

    # ✅ Encode Categorical Variables
    label_encoders = {}
    for col in categorical_columns:
        label_encoders[col] = SafeLabelEncoder()
        X[col] = label_encoders[col].fit_transform(X[col])

    # ✅ Encode the Target Variable (Missing Player)
    label_encoders[target_column] = SafeLabelEncoder()
    label_encoders[target_column].fit(all_players)  # Fit on combined player list
    y = label_encoders[target_column].transform(y)  # Transform training labels
    
    # ✅ Split the data into training and testing sets (with reference data)
    try:
        X_train, X_test, y_train, y_test, game_train, game_test, home_team_train, home_team_test, away_team_train, away_team_test = train_test_split(
            X, y, game_ids, original_home_teams, original_away_teams, test_size=0.2, random_state=42, stratify=y
        )
    except ValueError:
        # If stratify fails (e.g., too many classes), try without stratification
        X_train, X_test, y_train, y_test, game_train, game_test, home_team_train, home_team_test, away_team_train, away_team_test = train_test_split(
            X, y, game_ids, original_home_teams, original_away_teams, test_size=0.2, random_state=42
        )

    # ✅ Scale numerical features
    scaler = StandardScaler()
    if numerical_columns:
        X_train[numerical_columns] = scaler.fit_transform(X_train[numerical_columns])
        X_test[numerical_columns] = scaler.transform(X_test[numerical_columns])

    rf_model = RandomForestClassifier(
        n_estimators=225,
        max_depth=25,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        class_weight='balanced',
        n_jobs=-1,
        random_state=42
    )


    
    # Train the model
    rf_model.fit(X_train, y_train)
    rf_predictions = rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, rf_predictions)
    
    
    # ✅ Create results DataFrame
    results = pd.DataFrame()
    results['game'] = game_test
    results['season'] = X_test['season'].values if 'season' in X_test.columns else None
    results['home_team'] = home_team_test  # original team names
    results['away_team'] = away_team_test  # original team names
    results['actual_player'] = label_encoders[target_column].inverse_transform(y_test)
    results['predicted_player'] = label_encoders[target_column].inverse_transform(rf_predictions)
    
    # ✅ Calculate feature importances
    feature_importances = pd.DataFrame(
        rf_model.feature_importances_,
        index=input_features,
        columns=['importance']
    ).sort_values('importance', ascending=False)
    
    return rf_model, label_encoders, scaler, results, feature_importances, accuracy

def save_results(results, importances, position):
    """
    Saves results to an Excel file, overwriting existing files.
    """
    filename = f'nba_predictions_results_{position}.xlsx'
    try:
        with pd.ExcelWriter(filename, mode='w') as writer:
            results.to_excel(writer, sheet_name='Predictions', index=False)
            importances.to_excel(writer, sheet_name='Feature Importance')
        print(f"Results saved successfully to {filename}")
        return True
    except Exception as e:
        print(f"Error saving results for {position}: {str(e)}")
        return False

def analyze_predictions(results_all):
    """
    Analyze prediction results to find patterns in errors.
    """
    all_results = pd.concat(results_all.values())
    correct_predictions = all_results['actual_player'] == all_results['predicted_player']
    accuracy = correct_predictions.mean()
    
    # Analysis by team
    team_accuracy = all_results.groupby('home_team').apply(
        lambda x: (x['actual_player'] == x['predicted_player']).mean()
    ).sort_values()
    
    # Analysis by season if available
    season_accuracy = None
    if 'season' in all_results.columns:
        season_accuracy = all_results.groupby('season').apply(
            lambda x: (x['actual_player'] == x['predicted_player']).mean()
        ).sort_values()
    
    # Most common misclassifications
    error_cases = all_results[~correct_predictions].copy()
    error_pairs = error_cases.groupby(['actual_player', 'predicted_player']).size().sort_values(ascending=False)
    
    return {
        'overall_accuracy': accuracy,
        'team_accuracy': team_accuracy,
        'season_accuracy': season_accuracy,
        'common_errors': error_pairs.head(10)
    }

def generate_player_stats(combined_data):
    """
    Generate basic player statistics from the lineup data.
    Returns a dataframe with player frequency by position and team.
    """
    player_stats_list = []
    
    # Process all positions
    for position in [f'home_{i}' for i in range(5)] + [f'away_{i}' for i in range(5)]:
        pos_number = int(position.split('_')[1])
        team_type = position.split('_')[0]  # home or away
        
        # Count player occurrences by position
        player_counts = combined_data.groupby([position, f'{team_type}_team', 'season']).size().reset_index()
        player_counts.columns = ['player', 'team', 'season', 'frequency']
        player_counts['position'] = pos_number
        
        player_stats_list.append(player_counts)
    
    # Combine all position data
    position_stats = pd.concat(player_stats_list, ignore_index=True)
    
    # Calculate position frequency
    position_stats_agg = position_stats.groupby(['player', 'team', 'season']).agg({
        'frequency': 'sum',
        'position': 'mean'  # Average position (0-4)
    }).reset_index()
    
    # Calculate historical player stats
    historical_stats = position_stats.groupby(['player']).agg({
        'frequency': 'sum',
        'position': 'mean'
    }).reset_index()
    
    return position_stats_agg, historical_stats

def generate_team_stats(combined_data):
    """
    Generate team statistics from the lineup data.
    """
    # Home team stats
    home_lineups = combined_data.groupby(['home_team', 'season']).size().reset_index(name='lineup_count')
    home_lineups['team'] = home_lineups['home_team']
    
    # Away team stats
    away_lineups = combined_data.groupby(['away_team', 'season']).size().reset_index(name='away_lineup_count')
    away_lineups['team'] = away_lineups['away_team']
    
    # Merge home and away stats
    team_stats = pd.merge(home_lineups, away_lineups[['team', 'season', 'away_lineup_count']], 
                          on=['team', 'season'], how='outer').fillna(0)
    
    # Calculate player usage stats by team
    for position in range(5):
        # Calculate versatility (number of unique players used)
        home_pos = f'home_{position}'
        away_pos = f'away_{position}'
        
        home_versatility = combined_data.groupby(['home_team', 'season'])[home_pos].nunique().reset_index()
        home_versatility.columns = ['team', 'season', f'h{position}_versatility']
        
        away_versatility = combined_data.groupby(['away_team', 'season'])[away_pos].nunique().reset_index()
        away_versatility.columns = ['team', 'season', f'a{position}_versatility']
        
        team_stats = pd.merge(team_stats, home_versatility, on=['team', 'season'], how='left')
        team_stats = pd.merge(team_stats, away_versatility, on=['team', 'season'], how='left')
    
    # Fill NaN values
    team_stats = team_stats.fillna(0)
    
    return team_stats

if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore')

    year_to_test = config["year_to_test"] ## Replace with the desired year

    

    test_file = config["test_file"]

    if os.path.exists(test_file):
        test_data = pd.read_csv(test_file)
        print(f"Test file {test_file} successfully loaded")
    
    else:
        print(f"❌ Warning: Test file not found for year {config['year_to_test']}. Ensure it exists in {config['test_dir']}.")


    # ----- Training Phase -----
    # Load training data from CSV files in the 'data' directory
    data_dir = 'data'
    all_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.csv')]

    if year_to_test == 2007:
        training_years = [2007]
    elif year_to_test == 2016:
        training_years = [2015]
    elif year_to_test == 2008:
        training_years = [2007,2008]
    else:
        training_years = [year_to_test - 2, year_to_test - 1, year_to_test]
    
    selected_files = [file for file in all_files if any(f"matchups-{year}.csv" in file for year in training_years)]

    data_frames = []
    
    for file in selected_files:
        try:
            df = pd.read_csv(file)
            data_frames.append(df)
        except Exception as e:
            print(f"Error loading {file}: {e}")
    
    if data_frames:
        combined_data = pd.concat(data_frames, ignore_index=True).dropna()
        combined_data.drop_duplicates(inplace=True)
    else:
        print("No training data found for the selected year range.")
        combined_data = pd.DataFrame()  


    

    # Filter for winning home team samples (if the 'outcome' column exists)
    if 'outcome' in combined_data.columns:
        combined_data = combined_data[combined_data['outcome'] == 1]

    print(f"\nNumber of winning home team samples: {len(combined_data)}")
    print(f"Seasons covered: {combined_data['season'].unique().tolist()}")

    # Generate player and team statistics
    print("Generating player and team statistics...")
    position_stats, historical_stats = generate_player_stats(combined_data)
    team_stats = generate_team_stats(combined_data)

    # Dictionary to store trained models and results
    models = {}
    results_all = {}

    # ✅ Testing on the fly - No model storage
    test_results = {}

    for position in [f'home_{i}' for i in range(5)]:
        print(f"\n🔄 Training and testing model for missing position: {position}")

        # Create datasets for training
        X_train, y_train, input_features, target_column, game_ids, original_home_teams, original_away_teams = create_position_dataset(
            combined_data, position, player_stats=position_stats, team_stats=team_stats
        )

        # ✅ Train model & encode on the fly (no saving/loading)
        model, encoders, scaler, results, importances, accuracy = train_position_model(
            X_train, y_train, input_features, target_column, game_ids, original_home_teams, original_away_teams, test_data
        )

        print(f"✅ Model Accuracy for {position}: {accuracy:.4f}")

        # ✅ Now perform predictions on test data
        test_subset = test_data[test_data[position] == '?'].copy()

        if test_subset.empty:
            print(f"ℹ️ No test samples with missing {position}. Skipping...")
            continue

        # Track original indices
        test_subset['original_index'] = test_subset.index

        # Create test dataset
        X_test, _, _, _, _, _, _ = create_position_dataset(
            test_subset, position, player_stats=position_stats, team_stats=team_stats
        )

        # Ensure test data uses only the trained features
        common_features = [col for col in input_features if col in X_test.columns]
        X_test = X_test[common_features].fillna(0)

        # Identify categorical and numerical columns
        categorical_columns = ['home_team', 'away_team', 'matchup']
        categorical_columns.extend([col for col in X_test.columns if X_test[col].dtype == 'object'])
        numerical_columns = [col for col in X_test.columns if col not in categorical_columns]

        # Encode categorical features using trained encoders
        for col in categorical_columns:
            if col in encoders:
                X_test[col] = X_test[col].astype(str).map(encoders[col].mapping_).fillna(-1).astype(int)

        # Scale numerical features
        if numerical_columns:
            X_test[numerical_columns] = scaler.transform(X_test[numerical_columns])

        # Predict missing players
        predictions = model.predict(X_test)
        predicted_players = encoders[position].inverse_transform(predictions)

        # Store predictions
        test_subset[f'predicted_{position}'] = predicted_players
        test_results[position] = test_subset

    # ✅ Combine all test results
    if test_results:
        combined_test_results = pd.concat(test_results.values(), ignore_index=True)
        combined_test_results.sort_values('original_index', inplace=True)

        # Extract predicted players
        predicted_columns = [f'predicted_home_{i}' for i in range(5)]
        combined_test_results['predicted_player'] = combined_test_results[predicted_columns].apply(
            lambda row: next((val for val in row if pd.notnull(val)), None), axis=1
        )

   
        

        labels_file = config["labels_file"]

        if os.path.exists(labels_file):
            labels_df = pd.read_csv(labels_file)
            print(f"Successfully loaded label file {labels_file}")

        else:
            print(f"❌ Warning: Label file not found for year {config['year_to_test']}. Ensure it exists in {config['labels_dir']}.")

            labels_df = pd.DataFrame()

        combined_test_results['actual_player'] = labels_df['removed_value'].values

        # ✅ Compute accuracy
        test_accuracy = accuracy_score(
            combined_test_results['actual_player'], 
            combined_test_results['predicted_player']
        )
        print(f"\n🎯 Final Test Accuracy: {test_accuracy:.4f}")
        

        # Compute the weighted F1-score (handles class imbalance)
        f1 = f1_score(combined_test_results['actual_player'], combined_test_results['predicted_player'], average='weighted')

        print(f"Final F1-score : {f1}")

    

        # ✅ Save results to Excel
        save_path = os.path.join(config["save_path"], f'NBA_test_predictions_{config["year_to_test"]}.xlsx')

        # ✅ Save results to the specified folder
        combined_test_results.to_excel(save_path, index=False)
        print(f"✅ Test results saved to: {save_path}")

    else:
        print("❌ No test predictions were made.")
