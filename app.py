from flask import Flask, request, jsonify, render_template, g
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
import logging
import firebase_admin
from firebase_admin import credentials, db
import json
from datetime import datetime
import sqlite3

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
CORS(app)

cred = credentials.Certificate("rf-bioflok-firebase-adminsdk-fbsvc-617560ca39.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://rf-bioflok-default-rtdb.asia-southeast1.firebasedatabase.app/sensor.json'
})


DATABASE = 'log_data.db'

def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(DATABASE)
    return db

@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()

def init_db():
    with app.app_context():
        db = get_db()
        db.execute('''
            CREATE TABLE IF NOT EXISTS logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                suhu REAL,
                ph REAL,
                kekeruhan REAL,
                status TEXT
            )
        ''')
        db.commit()
        logging.info("Tabel logs siap digunakan.")


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

        # Simpan ke SQLite
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        db_conn = get_db()
        db_conn.execute(
            'INSERT INTO logs (timestamp, suhu, ph, kekeruhan, status) VALUES (?, ?, ?, ?, ?)',
            (timestamp, data['suhu'], data['ph'], data['kekeruhan'], str(classification_prediction))
        )
        db_conn.commit()

        result = {
            'raw_data': data,
            'classification': str(classification_prediction),
            'regression': round(float(regression_prediction), 2),
            'timestamp': timestamp
        }

        return jsonify(result)

    except Exception as e:
        logging.error(f"Error saat mengambil data dari Firebase: {e}", exc_info=True)
        return jsonify({'error': 'Gagal ambil data dari Firebase'}), 500

@app.route('/firebase-logs', methods=['GET'])
def get_firebase_logs_sqlite():
    try:
        db_conn = get_db()
        cursor = db_conn.execute(
            'SELECT timestamp, suhu, ph, kekeruhan, status FROM logs ORDER BY timestamp DESC LIMIT 50'
        )
        rows = cursor.fetchall()

        logs = []
        for row in rows:
            try:
                tanggal, waktu = row[0].split(' ')
                logs.append({
                    'date': tanggal,
                    'time': waktu[:5],
                    'temperature': row[1],
                    'ph': row[2],
                    'turbidity': row[3],
                    'status': row[4]
                })
            except Exception as e:
                logging.warning(f"Kesalahan format data log: {row} -> {e}")

        return jsonify(logs)

    except Exception as e:
        logging.error(f"Gagal ambil log dari SQLite: {e}", exc_info=True)
        return jsonify({'error': 'Gagal ambil data log dari database'}), 500


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



@app.route('/index')
def index_page():
    return render_template('index.html')

@app.route('/log-data')
def log_data_page():
    return render_template('log data.html')


if __name__ == '__main__':
    init_db()
    app.run(debug=True)
    