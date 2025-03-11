
from app import app  # Ganti `myapp` dengan nama aplikasi Flask-mu

if __name__ == "__main__":
    from waitress import serve
    serve(app, host="0.0.0.0", port=8000)
