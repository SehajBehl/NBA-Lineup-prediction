# Configuration Guide for config.py

## Overview
The `config.py` file is used to store user-defined configuration variables for the NBA prediction model. It allows users to easily modify key parameters such as the year to test, file paths, and directories without altering the main model script.

## Configuration Variables
The `config.py` file contains a dictionary named `config` that holds the following key-value pairs:

```python
config = {
    "year_to_test": 2007,  # Specify the year for which you want to test the model
    "labels_dir": r"<REPLACE WITH YOUR LOCAL DIRECTORY PATH>",  # Directory containing label files (e.g. NBA_LABELS )
    "test_dir": r"<REPLACE WITH YOUR LOCAL DIRECTORY PATH>",  # Directory containing test data files (e.g. NBA_Test)
    "save_path": r"<REPLACE WITH YOUR LOCAL DIRECTORY PATH>",  # Directory to save prediction results (e.g. NBA_predictions)
}
```

## How to Use
1. **Set the Year to Test**  
   Modify the `year_to_test` value to specify the year of test data you want to use.
   - Example: To test the model for 2010, change it to:
   
     ```python
     "year_to_test": 2010,
     ```

2. **Ensure Directories Exist**  
   - The `labels_dir` should contain CSV files in the format `NBA_labels_{year}.csv` (e.g., `NBA_labels_2010.csv`).
   - The `test_dir` should contain CSV files in the format `NBA_test_{year}.csv` (e.g., `NBA_test_2010.csv`).
   - The `save_path` should be a valid directory where the model will store its output.

3. **Save the Config File**  
   After making changes, save the `config.py` file. There is **no need to run it** separately. The main model script will import it when executed.

## Expected File Structure
Make sure your project follows this structure for correct execution:

```
NBA_Prediction/
│── config.py
│── model_script.py
│── NBA_labels/
│   ├── NBA_labels_2007.csv
│   ├── NBA_labels_2008.csv
│── NBA_test/
│   ├── NBA_test_2007.csv
│   ├── NBA_test_2008.csv
│── NBA_predictions/
```

## Common Issues & Fixes
✅ **Issue: Test file not found**  
   **Fix:** Ensure the file `NBA_test_{year_to_test}.csv` exists in the `NBA_test` directory.

✅ **Issue: Label file not found**  
   **Fix:** Ensure the file `NBA_labels_{year_to_test}.csv` exists in the `NBA_labels` directory.

✅ **Issue: Results not saving**  
   **Fix:** Ensure `save_path` points to a valid directory.

## Additional Notes
- The `config.py` file is designed to make the model adaptable to different test years without modifying the main script.
- If you want to use a different file structure, update the corresponding paths in `config.py`.
- Make sure you save the `config.py` file before running `RandomForest.py`

For any issues, double-check the directory paths and filenames. 

