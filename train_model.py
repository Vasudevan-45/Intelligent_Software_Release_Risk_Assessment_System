import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ----------------------------
# Load Dataset
# ----------------------------
df = pd.read_csv("software_release_risk_dataset.csv")

# ----------------------------
# Encode Target
# ----------------------------
label_encoder = LabelEncoder()
df["Risk_Level"] = label_encoder.fit_transform(df["Risk_Level"])

# ----------------------------
# Features and Target
# ----------------------------
X = df.drop(["Risk_Level", "Release_ID"], axis=1)
y = df["Risk_Level"]

# ----------------------------
# Train-Test Split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ----------------------------
# Train Random Forest
# ----------------------------
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

# ----------------------------
# Evaluate
# ----------------------------
y_pred = model.predict(X_test)

print("=" * 50)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("=" * 50)
print(classification_report(y_test, y_pred))

# ----------------------------
# Save Model
# ----------------------------
joblib.dump(model, "release_risk_model.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")

print("\nModel Saved Successfully!")
print("release_risk_model.pkl created")
print("label_encoder.pkl created")
