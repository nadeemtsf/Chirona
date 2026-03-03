# training script
import pickle
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(BASE_DIR, "data", "landmarks.pickle")

if not os.path.exists(data_path):
    print("Error: Run python data/extract_landmarks.py first.")
    exit(1)

with open(data_path, "rb") as f:
    dataset = pickle.load(f)

X = np.array(dataset["data"])
y = np.array(dataset["labels"])

# stratify ensures equal class distribution in both train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Test Accuracy: {accuracy:.2f}")

# save to models/saved/model_rf.p — loaded by sign_classifier at runtime
model_dir = os.path.join(BASE_DIR, "models", "saved")
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, "model_rf.p")

with open(model_path, "wb") as f:
    pickle.dump(model, f)

print(f"Model saved to {model_path}")