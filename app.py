# app.py
from flask import Flask, request, jsonify, render_template
import os, tempfile
import soundfile as sf
from pydub import AudioSegment
from audio_utils import load_audio, extract_features, compute_risk_from_features

app = Flask(__name__)
RISK_THRESHOLD = 0.45

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    if "audio" not in request.files:
        return jsonify({"error": "No audio uploaded"}), 400

    f = request.files["audio"]

    # Save raw uploaded file
    raw_path = tempfile.NamedTemporaryFile(delete=False, suffix=".tmp").name
    f.save(raw_path)

    try:
        # Convert ANY audio format -> WAV
        audio = AudioSegment.from_file(raw_path)
        wav_path = raw_path + ".wav"
        audio.export(wav_path, format="wav")
    except Exception as e:
        return jsonify({"error": f"Audio conversion failed: {e}"}), 500

    try:
        # Load WAV for analysis
        y, sr = load_audio(wav_path)
        feats = extract_features(y, sr)
        risk = compute_risk_from_features(feats)

        return jsonify({
            "risk_score": round(risk, 4),
            "threshold": RISK_THRESHOLD,
            "features": feats,
        })
    finally:
        # Cleanup temp files
        try: os.unlink(raw_path)
        except: pass
        try: os.unlink(wav_path)
        except: pass

if __name__ == "__main__":
    app.run(debug=True)
