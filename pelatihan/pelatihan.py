# Import library
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Buat data dummy
np.random.seed(42)
data_size = 100
temperature = np.random.uniform(20, 30, data_size)
ph = np.random.uniform(6.5, 8.5, data_size)
turbidity = np.random.uniform(0, 10, data_size)
label = np.random.choice(['Keruh', 'Jernih'], data_size)

# Buat DataFrame
df = pd.DataFrame({
    'temperature': temperature,
    'ph': ph,
    'turbidity': turbidity,
    'label': label
})

# Pisahkan fitur dan label
X = df[['temperature', 'ph', 'turbidity']]
y = df['label']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Buat model Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluasi
y_pred = model.predict(X_test)
print("Akurasi:", accuracy_score(y_test, y_pred))

# Simpan model
joblib.dump(model, 'random_forest_model.pkl')
print("Model disimpan sebagai 'random_forest_model.pkl'")
