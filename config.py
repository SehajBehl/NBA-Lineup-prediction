import os

# ✅ User-defined variables
config = {
    "year_to_test": 2014,  # Replace with the desired test year
    "labels_dir": r"",  # Directory containing label files
    "test_dir": r"",  # Directory containing test data files
    "save_path": r"",  # Folder to save results
}

config["test_file"] = os.path.join(config["test_dir"], f'NBA_test_{config["year_to_test"]}.csv')
config["labels_file"] = os.path.join(config["labels_dir"], f'NBA_labels_{config["year_to_test"]}.csv')
config["save_file"] = os.path.join(config["save_path"], f'NBA_test_predictions_{config["year_to_test"]}.xlsx')
