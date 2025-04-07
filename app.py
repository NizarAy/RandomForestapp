from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd
import logging
import firebase_admin
from firebase_admin import credentials, db
import json
from datetime import datetime

# Load model klasifikasi
try:
    classification_model = joblib.load('model/random_forest_model.pkl')
    regression_model = joblib.load('model/random_forest_regressor_model.pkl')
    print("Model regresi berhasil dimuat.")
    print("Model klasifikasi berhasil dimuat.")
except Exception as e:
    print(f"Error saat memuat model klasifikasi: {e}")
    exit()

# Konfigurasi logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Inisialisasi Flask
app = Flask(__name__)

cred = credentials.Certificate("rf-bioflok-firebase-adminsdk-fbsvc-617560ca39.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://rf-bioflok-default-rtdb.asia-southeast1.firebasedatabase.app/sensor.json'
})


# Endpoint untuk halaman utama
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        try:
            # Ambil data dari form
            temperature = float(request.form.get('temperature'))
            ph = float(request.form.get('ph'))
            turbidity = float(request.form.get('turbidity'))

            # Buat DataFrame untuk prediksi
            features = pd.DataFrame([[temperature, ph, turbidity]],
                                   columns=['temperature', 'ph', 'turbidity'])

            # Lakukan prediksi regresi
            regression_prediction = regression_model.predict(features)[0]
            regression_result = {'prediction': round(float(regression_prediction), 2)}

            # Lakukan prediksi klasifikasi
            classification_prediction = classification_model.predict(features)[0]
            classification_result = {'label': str(classification_prediction)}

            # Gabungkan hasil prediksi
            result = {
                'regression': regression_result,
                'classification': classification_result
            }

            return jsonify(result)

        except Exception as e:
            print(f"Error pada form submission: {e}")
            return jsonify({'error': 'Terjadi kesalahan saat memproses data form.'}), 500

    # Jika metode GET, tampilkan halaman HTML
    return render_template('index.html')



@app.route('/firebase-data', methods=['GET'])
def get_firebase_data():
    try:
        ref = db.reference('sensor')
        data = ref.get()

        if not data:
            return jsonify({'error': 'Data kosong.'}), 404

        features = pd.DataFrame([[data['suhu'], data['ph'], data['kekeruhan']]],
                                columns=['temperature', 'ph', 'turbidity'])

        classification_prediction = classification_model.predict(features)[0]
        regression_prediction = regression_model.predict(features)[0]

        result = {
            'raw_data': data,
            'classification': str(classification_prediction),
            'regression': round(float(regression_prediction), 2),
            'timestamp': datetime.utcnow().isoformat()
        }

        print("Response to frontend:", json.dumps(result, indent=2))  # debug ke terminal
        return jsonify(result)

    except Exception as e:
        logging.error(f"Error saat mengambil data dari Firebase: {e}", exc_info=True)
        return jsonify({'error': 'Gagal ambil data dari Firebase'}), 500

# Endpoint untuk prediksi klasifikasi
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        print(f"Data diterima untuk klasifikasi: {data}")

        required_keys = ['temperature', 'ph', 'turbidity']
        if not all(key in data for key in required_keys):
            return jsonify({'error': 'Data tidak lengkap. Harus berisi temperature, ph, dan turbidity.'}), 400

        features = pd.DataFrame([[data['temperature'], data['ph'], data['turbidity']]],
                               columns=['temperature', 'ph', 'turbidity'])

        prediction = classification_model.predict(features)[0]
        print(f"Hasil prediksi klasifikasi: {prediction}")

        return jsonify({'label': str(prediction)})

    except Exception as e:
        print(f"Error pada prediksi klasifikasi: {e}")
        return jsonify({'error': 'Terjadi kesalahan saat memproses prediksi.'}), 500

# Endpoint untuk prediksi regresi
@app.route('/predict-regression', methods=['POST'])
def predict_regression():
    try:
        data = request.get_json()
        print(f"Data diterima untuk regresi: {data}")

        required_keys = ['temperature', 'ph', 'turbidity']
        if not all(key in data for key in required_keys):
            return jsonify({'error': 'Data tidak lengkap. Harus berisi temperature, ph, dan turbidity.'}), 400

        features = pd.DataFrame([[data['temperature'], data['ph'], data['turbidity']]],
                               columns=['temperature', 'ph', 'turbidity'])

        prediction = regression_model.predict(features)[0]
        print(f"Hasil prediksi regresi: {prediction}")

        return jsonify({'turbidity_level': round(float(prediction), 2)})

    except Exception as e:
        print(f"Error pada prediksi regresi: {e}")
        return jsonify({'error': 'Terjadi kesalahan saat memproses prediksi regresi.'}), 500

# Endpoint untuk prediksi 30 hari ke depan menggunakan klasifikasi
@app.route('/predict-30-days', methods=['POST'])
def predict_30_days():
    try:
        data = request.get_json()
        logging.info(f"Data diterima untuk prediksi 30 hari klasifikasi: {data}")

        required_keys = ['temperature', 'ph', 'turbidity']
        if not all(key in data for key in required_keys):
            logging.error("Data tidak lengkap. Harus berisi temperature, ph, dan turbidity.")
            return jsonify({'error': 'Data tidak lengkap. Harus berisi temperature, ph, dan turbidity.'}), 400

        predictions = []
        for i in range(30):
            temp = data['temperature'] + np.random.uniform(-1, 1)
            ph = data['ph'] + np.random.uniform(-0.2, 0.2)
            turbidity = data['turbidity'] + np.random.uniform(-0.5, 0.5)

            features = pd.DataFrame([[temp, ph, turbidity]],
                                   columns=['temperature', 'ph', 'turbidity'])
            prediction = classification_model.predict(features)[0]
            predictions.append({'day': i + 1, 'label': str(prediction)})

        logging.info(f"Hasil prediksi 30 hari klasifikasi: {predictions}")
        return jsonify(predictions)

    except Exception as e:
        logging.error(f"Error pada prediksi 30 hari klasifikasi: {e}", exc_info=True)
        return jsonify({'error': 'Terjadi kesalahan saat memproses prediksi 30 hari.'}), 500

@app.route('/predict-30-days-regression', methods=['POST'])
def predict_30_days_regression():
    try:
        data = request.get_json()
        logging.info(f"Data diterima untuk prediksi 30 hari regresi: {data}")

        required_keys = ['temperature', 'ph', 'turbidity']
        if not all(key in data for key in required_keys):
            logging.error("Data tidak lengkap. Harus berisi temperature, ph, dan turbidity.")
            return jsonify({'error': 'Data tidak lengkap. Harus berisi temperature, ph, dan turbidity.'}), 400

        predictions = []
        for i in range(30):
            temp = data['temperature'] + np.random.uniform(-1, 1)
            ph = data['ph'] + np.random.uniform(-0.2, 0.2)
            turbidity = data['turbidity'] + np.random.uniform(-0.5, 0.5)

            features = pd.DataFrame([[temp, ph, turbidity]],
                                  columns=['temperature', 'ph', 'turbidity'])
            prediction = regression_model.predict(features)[0]
            predictions.append({'day': i + 1, 'turbidity_level': round(float(prediction), 2)})

        logging.info(f"Hasil prediksi 30 hari regresi: {predictions}")
        return jsonify(predictions)

    except Exception as e:
        logging.error(f"Error pada prediksi 30 hari regresi: {e}", exc_info=True)
        return jsonify({'error': 'Terjadi kesalahan saat memproses prediksi 30 hari regresi.'}), 500

if __name__ == '__main__':
    app.run(debug=True)