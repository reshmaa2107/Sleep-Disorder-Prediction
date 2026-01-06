import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# 1. Load Data
print("📊 Loading Dataset...")
df = pd.read_csv('sleep_disorder_dataset.csv')

# Encoding
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])
df['Occupation'] = le.fit_transform(df['Occupation'])

X = df.drop('Disorder', axis=1)
y = df['Disorder']

# Split & Scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 2. Define 3 Different Neural Networks
models = {
    "Multi layer perceptron": MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=2000, random_state=42), # BEST
    "Shallow Network (1 Layer)": MLPClassifier(hidden_layer_sizes=(5,), max_iter=2000, random_state=42), # Too Simple
    "Deep Narrow Network": MLPClassifier(hidden_layer_sizes=(4, 4, 4, 4), max_iter=2000, random_state=42) # Hard to train
}

results = {}

print("🚀 Training Architectures...")
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, preds)
    results[name] = acc * 100
    print(f"   -> {name}: {acc*100:.2f}%")

# --- GRAPH: ARCHITECTURE COMPARISON ---
plt.figure(figsize=(10, 6))
# Colors: Purple for Winner, Gray for Losers
colors = ['#8B5CF6' if "Multi" in x else '#CBD5E1' for x in results.keys()]

bars = plt.bar(results.keys(), results.values(), color=colors)
plt.title("Neural Network Architecture Optimization", fontsize=16, fontweight='bold')
plt.ylabel("Accuracy (%)", fontsize=12)
plt.ylim(60, 100) # Zoom in to show the gap clearly

# Add numbers
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.5, f"{yval:.1f}%", ha='center', fontweight='bold')

plt.savefig('architecture_comparison.png')
print("\n✅ Graph Saved: architecture_comparison.png")
plt.show()