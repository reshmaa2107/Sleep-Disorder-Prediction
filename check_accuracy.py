import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder

# 1. Load the data again
df = pd.read_csv('sleep_disorder_dataset.csv')

# 2. Preprocess (Convert text to numbers)
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])
df['Occupation'] = le.fit_transform(df['Occupation'])

X = df.drop('Disorder', axis=1)
y = df['Disorder']

# 3. Split into Training (80%) and Testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Scale inputs
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Train a temporary model to check score
mlp = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=1000, random_state=42)
mlp.fit(X_train_scaled, y_train)

# 6. Predict and Print Accuracy
y_pred = mlp.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)

print("---------------------------------------")
print(f"📊 MODEL ACCURACY: {acc * 100:.2f}%")
print("---------------------------------------")
print("Detailed Report:")
print(classification_report(y_test, y_pred))