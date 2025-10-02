import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
csv_path = 'audio_features.csv'
df = pd.read_csv(csv_path)

# Visualize class distribution
plt.figure(figsize=(5,3))
sns.countplot(x='label', data=df)
plt.title('Class Distribution (0=Real, 1=Fake)')
plt.show()

# Visualize feature distributions for a few features
feature_cols = df.columns[:-1]
for col in feature_cols[:5]:  # Show first 5 features
    plt.figure(figsize=(5,3))
    sns.histplot(data=df, x=col, hue='label', kde=True, element='step', stat='density')
    plt.title(f'Feature: {col} by Class')
    plt.show()

# Print class counts
print('Class counts:')
print(df['label'].value_counts())
