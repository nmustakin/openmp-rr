import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
import sys

# Assuming the first command line argument is the path to the CSV file
if len(sys.argv) < 2:
    print("Usage: python script.py <path_to_csv_file>")
    sys.exit(1)

csv_file_path = sys.argv[1]
kernel_name = csv_file_path.split('/')[-1].split('___')[1].rsplit('.', 1)[0]

# Loading the CSV data from the file specified in the command line
data = pd.read_csv(csv_file_path)

# Creating plots and saving them as images
fig, axs = plt.subplots(2, 2, figsize=(15, 10))

sns.scatterplot(data=data, x='MaxThreads', y='Speedup', ax=axs[0, 0])
axs[0, 0].set_title('MaxThreads vs Speedup')

sns.scatterplot(data=data, x='MinTeams', y='Speedup', ax=axs[0, 1])
axs[0, 1].set_title('MinTeams vs Speedup')

sns.scatterplot(data=data, x='NumThreads', y='Speedup', ax=axs[1, 0])
axs[1, 0].set_title('NumThreads vs Speedup')

sns.scatterplot(data=data, x='NumTeams', y='Speedup', ax=axs[1, 1])
axs[1, 1].set_title('NumTeams vs Speedup')
plt.savefig('Parameter_vs_Speedup.png', bbox_inches='tight')
plt.close()

# Creating plots and saving them as images
fig, axs = plt.subplots(2, 2, figsize=(15, 10))

sns.scatterplot(data=data, x='MaxThreads', y='Energy_Improvment', ax=axs[0, 0])
axs[0, 0].set_title('MaxThreads vs Energy_Improvment')

sns.scatterplot(data=data, x='MinTeams', y='Energy_Improvment', ax=axs[0, 1])
axs[0, 1].set_title('MinTeams vs Energy_Improvment')

sns.scatterplot(data=data, x='NumThreads', y='Energy_Improvment', ax=axs[1, 0])
axs[1, 0].set_title('NumThreads vs Energy_Improvment')

sns.scatterplot(data=data, x='NumTeams', y='Energy_Improvment', ax=axs[1, 1])
axs[1, 1].set_title('NumTeams vs Energy_Improvment')
plt.savefig('Parameter_vs_Energy_Improvment.png', bbox_inches='tight')
plt.close()

data['Speedup_Energy_Sum'] = data['Speedup'] + data['Energy_Improvment']
top_30_rows = data.sort_values(by='Speedup_Energy_Sum', ascending=False).head(30)

plt.figure(figsize=(15, 10))

for _, row in top_30_rows.iterrows():
    plt.scatter(row['Speedup'], row['Energy_Improvment'], color='blue')
    plt.text(row['Speedup'], row['Energy_Improvment'],
             f"({row['MaxThreads']},{row['MinTeams']},{row['NumThreads']},{row['NumTeams']})",
             fontsize=10, ha='left')

plt.xlabel('Speedup')
plt.ylabel('Energy Improvement')
plt.title('Speedup vs Energy Improvement for Top Configurations')
plt.tight_layout()
plt.savefig(f'Top30-Speedup_vs_Gain_{kernel_name}.png', bbox_inches='tight')
plt.close()

print(top_30_rows.to_csv(index=False))
