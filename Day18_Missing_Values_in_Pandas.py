import pandas as pd
import numpy as np

# Load Titanic dataset
df = pd.read_csv('titanic.csv')

# 1. Missing-value summary
missing_summary = pd.DataFrame({
    'Missing_Count': df.isna().sum(),
    'Missing_Percentage': (df.isna().mean() * 100).round(2)
}).sort_values('Missing_Count', ascending=False)

print(missing_summary)

# 2. isna() and notna()
print("\nTotal missing values:", df.isna().sum().sum())
print("Missing Age:", df['age'].isna().sum())
print("Rows with Age available:")
print(df.loc[df['age'].notna(), ['name','age']].head())

# 3. Dropping
drop_age = df.dropna(subset=['age'])
drop_embarked = df.dropna(subset=['embarked'])

# 4. Filling
cleaned = df.copy()
cleaned['age'] = cleaned['age'].fillna(cleaned['age'].median())
cleaned['embarked'] = cleaned['embarked'].fillna(cleaned['embarked'].mode()[0])
cleaned['cabin'] = cleaned['cabin'].fillna('Unknown')

# 5. Comparison
comparison = pd.DataFrame({
    'Before': df.isna().sum(),
    'After': cleaned.isna().sum()
})
comparison['Filled_or_Remaining'] = comparison['Before'] - comparison['After']
print("\nBefore vs After:")
print(comparison)

# 6. Save outputs
cleaned.to_csv('titanic_cleaned.csv', index=False)
missing_summary.to_csv('missing_value_summary.csv')
print("\nCreated titanic_cleaned.csv and missing_value_summary.csv")
