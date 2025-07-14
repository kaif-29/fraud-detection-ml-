import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# --- Step 1: Load the Dataset ---
# Download the 'creditcard.csv' dataset (from Kaggle or similar) into your project folder.
# Example dataset: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
df = pd.read_csv("creditcard.csv")
print("Dataset loaded. Sample rows:")
print(df.head())

# --- Step 2: Check for Missing Values ---
print("\nMissing values by column:")
print(df.isnull().sum())

# --- Step 3: Encode Categorical Columns ---
# In the AmanXai article, 'type' was categorical, but in this dataset, there are no types.
# For demonstration, if you have a 'type' column, uncomment below:
# df['type_enc'] = df['type'].astype('category').cat.codes

# Here, we move ahead with numeric features only.

# --- Step 4: Prepare Features & Labels ---
features = df.drop(['Class'], axis=1)
labels = df['Class'].values  # 0 = not fraud, 1 = fraud

# --- Step 5: Split Data ---
X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.1, random_state=42
)
print(f"\nTraining samples: {len(X_train)}, Test samples: {len(X_test)}")

# --- Step 6: Train Decision Tree Model ---
model = DecisionTreeClassifier(max_depth=6, random_state=42)
model.fit(X_train, y_train)
print("\nModel training completed.")

# --- Step 7: Evaluate Model ---
y_pred = model.predict(X_test)
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# --- Optional: Predict Custom Sample ---
def predict_transaction(sample: dict):
    df_s = pd.DataFrame([sample])
    pred = model.predict(df_s)[0]
    return "Fraud" if pred == 1 else "Not Fraud"

print("\nExample prediction:",
      predict_transaction({col: X_test.iloc[0][col] for col in features.columns}))
