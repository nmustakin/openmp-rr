import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import sys

# Assuming the first command line argument is the path to the CSV file
if len(sys.argv) < 2:
    print("Usage: python script.py <path_to_csv_file>")
    sys.exit(1)

csv_file_path = sys.argv[1]
kernel_name = csv_file_path.split('/')[-1].split('___')[1].rsplit('.', 1)[0]

# Loading the CSV data from the file specified in the command line
data = pd.read_csv(csv_file_path)

#data['Speedup_Energy_Sum'] = data['Speedup'] + data['Energy_Improvment']
data['NumThreads_NumTeams'] = data['NumThreads'] * data['NumTeams']
data['MaxThreads_MinTeams'] = data['MaxThreads'] * data['MinTeams']
data['MaxThreads_NumThreads'] = data['MaxThreads'] * data['NumThreads']
data['MaxThreads_NumThreads'] = data['MaxThreads'] * data['NumTeams']
data['MinTeams_NumTeams'] = data['MinTeams'] * data['NumTeams']

# Preparing features and target variable
X = data[['MaxThreads', 'MinTeams', 'NumThreads', 'NumTeams', 
          'NumThreads_NumTeams', 'MaxThreads_MinTeams', 
          'MaxThreads_NumThreads', 'MinTeams_NumTeams']]
y = data['Energy_Improvment']

# Fitting a Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Getting feature importances
feature_importances = model.feature_importances_

# Displaying feature importances
features_df = pd.DataFrame({'Feature': X.columns, 'Importance_on_Energy': feature_importances}).sort_values(by='Feature', ascending=True)
print(features_df.to_csv(index=False))
