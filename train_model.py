import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
import joblib

df = pd.read_csv('sleep_disorder_dataset.csv')

le_gender = LabelEncoder()
df['Gender'] = le_gender.fit_transform(df['Gender'])
le_occup = LabelEncoder()
df['Occupation'] = le_occup.fit_transform(df['Occupation'])

X = df.drop('Disorder', axis=1)
y = df['Disorder']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# MLP Model
mlp = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=1000, random_state=42)
mlp.fit(X_train_scaled, y_train)

# Save files
joblib.dump(mlp, 'sleep_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(le_gender, 'le_gender.pkl')
joblib.dump(le_occup, 'le_occup.pkl')
print("✅ Model Trained!")
