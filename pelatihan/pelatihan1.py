# Import library
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib

# Buat data dummy
np.random.seed(42)
data_size = 100

# Fitur
temperature = np.random.uniform(20, 30, data_size)
ph = np.random.uniform(6.5, 8.5, data_size)
turbidity = np.random.uniform(0, 10, data_size)

# Target: kekeruhan (nilai antara 0 - 100)
turbidity_level = np.random.uniform(0, 100, data_size)

# Buat DataFrame
df = pd.DataFrame({
    'temperature': temperature,
    'ph': ph,
    'turbidity': turbidity,
    'turbidity_level': turbidity_level
})

# Pisahkan fitur dan target
X = df[['temperature', 'ph', 'turbidity']]
y = df['turbidity_level']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Buat model Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Latih model
model.fit(X_train, y_train)

# Evaluasi model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Simpan model
joblib.dump(model, 'random_forest_regressor_model.pkl')
print("Model disimpan sebagai 'random_forest_regressor_model.pkl'")
