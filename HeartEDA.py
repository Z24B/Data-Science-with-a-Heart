import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

# Load the dataset
file_path = "heart.csv"
df = pd.read_csv(file_path)

# Define numerical and categorical columns
categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal', 'target']
numerical_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

# Display basic dataset info
df_info = df.info()

# Check for missing values and duplicates
missing_values = df.isnull().sum()
duplicate_count = df.duplicated().sum()

# Drop duplicate records
df = df.drop_duplicates()

# Display dataset summary
df_description = df[numerical_cols].describe()
print("Dataset Summary:\n", df_description)
print("Missing Values:\n", missing_values)
print("Duplicate Count:\n", duplicate_count)

# Plot distributions of numerical features
df[numerical_cols].hist(figsize=(12, 8), bins=20, edgecolor="black")
plt.suptitle("Distribution of Numerical Features", fontsize=14)
plt.show()

# Count plots for categorical variables
plt.figure(figsize=(15, 10))
for i, col in enumerate(categorical_cols):
    plt.subplot(3, 3, i + 1)
    sns.countplot(x=df[col], hue=df[col], palette="viridis", legend=False)
    plt.title(f"Count Plot of {col}")
plt.tight_layout()
plt.show()

# Correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()

# Correlation with target variable
correlation_with_target = df.corr()["target"].sort_values(ascending=False)
print("Correlation with Target:\n", correlation_with_target)

# Feature Importance using Random Forest
X = df.drop(columns=["target"])
y = df["target"]

# Convert categorical variables into numerical format using one-hot encoding
X = pd.get_dummies(X, drop_first=True)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Get feature importance
feature_importance = pd.DataFrame({"Feature": X.columns, "Importance": model.feature_importances_})
feature_importance = feature_importance.sort_values(by="Importance", ascending=False)

print("Feature Importance:\n", feature_importance)
