"""Fix Root_Shoot_Ratio column in CSV file.

This script:
1. Replaces #VALUE! errors with NaN
2. Converts the column to numeric dtype
3. Creates a backup of the original file
"""

import pandas as pd
import numpy as np
from pathlib import Path
import shutil

# Read the CSV
csv_path = 'C:/Users/Elizabeth/Desktop/phenotyping/20251010_wheat_edpie/turface/Turface_all_traits_2024_RSR.csv'
df = pd.read_csv(csv_path)

print('Before cleaning:')
print(f'  Root_Shoot_Ratio dtype: {df["Root_Shoot_Ratio"].dtype}')
error_count = (df["Root_Shoot_Ratio"] == "#VALUE!").sum()
print(f'  #VALUE! errors: {error_count}')
valid_count = df["Root_Shoot_Ratio"].notna().sum() - error_count
print(f'  Valid values: {valid_count}')

# Replace #VALUE! with NaN
df['Root_Shoot_Ratio'] = df['Root_Shoot_Ratio'].replace('#VALUE!', np.nan)

# Convert to numeric
df['Root_Shoot_Ratio'] = pd.to_numeric(df['Root_Shoot_Ratio'], errors='coerce')

print('\nAfter cleaning:')
print(f'  Root_Shoot_Ratio dtype: {df["Root_Shoot_Ratio"].dtype}')
print(f'  NaN values: {df["Root_Shoot_Ratio"].isna().sum()}')
print(f'  Valid numeric values: {df["Root_Shoot_Ratio"].notna().sum()}')
if df["Root_Shoot_Ratio"].notna().any():
    print(f'  Min: {df["Root_Shoot_Ratio"].min():.4f}')
    print(f'  Max: {df["Root_Shoot_Ratio"].max():.4f}')
    print(f'  Mean: {df["Root_Shoot_Ratio"].mean():.4f}')

# Create backup
backup_path = csv_path.replace('.csv', '_backup.csv')
print(f'\nCreating backup: {Path(backup_path).name}')
shutil.copy2(csv_path, backup_path)

# Save fixed CSV
df.to_csv(csv_path, index=False)
print(f'✅ Saved fixed CSV to: {Path(csv_path).name}')
print('\n✨ The Root_Shoot_Ratio column is now properly numeric!')
print('   It will be detected by get_trait_columns() in your notebook.')
