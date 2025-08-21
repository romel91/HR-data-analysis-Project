# scripts/hr_analysis.py
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# 1. Load dataset
df = pd.read_csv("data/WA_Fn-UseC_-HR-Employee-Attrition.csv")

print("✅ Dataset loaded successfully")
print("Shape of dataset:", df.shape)
print("\nColumns:", df.columns.tolist())

# 2. Clean data
# Remove duplicates
df = df.drop_duplicates()

# Check for missing values
print("\nMissing values:\n", df.isnull().sum())

# Convert object columns to category
cat_cols = df.select_dtypes(include='object').columns
df[cat_cols] = df[cat_cols].apply(lambda x: x.astype('category'))

print("\nCategorical Columns:", cat_cols.tolist())

# 3. Encode categorical variables
encoder = LabelEncoder()
for col in df.select_dtypes(['category']).columns:
    df[col] = encoder.fit_transform(df[col])

# 4. Train/Test Split
X = df.drop('Attrition', axis=1)
y = df['Attrition']  # target column

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 5. Train Model (Random Forest)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 6. Predictions
y_pred = model.predict(X_test)

# 7. Metrics
print("\n✅ Model Training Completed")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 8. Example Plot (Attrition Distribution)
sns.countplot(x='Attrition', data=df)
plt.title("Employee Attrition Count")
plt.show()
