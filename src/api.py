from flask import Flask, request, jsonify
import pickle, os, time

MODEL_PATH = "/mnt/model/model.pkl"

app = Flask(__name__)
app.model = None
app.model_mtime = 0
VERSION = os.getenv("VERSION", "1.0.0")

def load_model():
    mtime = os.path.getmtime(MODEL_PATH)
    if mtime != app.model_mtime:
        with open(MODEL_PATH, "rb") as f:
            app.model = pickle.load(f)
        app.model_mtime = mtime

load_model()

@app.route("/api/recommend", methods=["POST"])
def recommend():
    load_model()
    data = request.get_json(force=True)
    songs = data["songs"]

    recommended = compute_recommendation(app.model, songs)

    return jsonify({
        "songs": recommended,
        "version": VERSION,
        "model_date": str(app.model_mtime)
    })
