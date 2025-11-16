# audio_utils.py
import numpy as np
import librosa

def load_audio(path, sr=16000):
    y, orig_sr = librosa.load(path, sr=sr, mono=True)
    return y, sr

def extract_features(y, sr=16000):
    features = {}

    hop_length = 512
    frame_length = 1024

    # RMS
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    features["rms_mean"] = float(np.mean(rms))
    features["rms_max"] = float(np.max(rms))

    # Spectral centroid
    spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
    features["spec_centroid_mean"] = float(np.mean(spec_centroid))
    features["spec_centroid_max"] = float(np.max(spec_centroid))

    # Zero crossing
    zcr = librosa.feature.zero_crossing_rate(
        y, frame_length=frame_length, hop_length=hop_length
    )[0]
    features["zcr_mean"] = float(np.mean(zcr))

    # MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop_length)
    for i in range(3):
        features[f"mfcc{i+1}_mean"] = float(np.mean(mfcc[i]))

    features["duration"] = float(len(y) / sr)
    return features

def compute_risk_from_features(feat):
    rms_max = feat.get("rms_max", 0.0)
    spec_centroid_max = feat.get("spec_centroid_max", 0.0)
    zcr_mean = feat.get("zcr_mean", 0.0)

    # Normalize
    def norm(x, low, high):
        if high <= low:
            return 0
        return max(0, min(1, (x - low) / (high - low)))

    rms_score = norm(rms_max, 0.02, 0.25)
    spec_score = norm(spec_centroid_max, 1000, 5000)
    zcr_score = norm(zcr_mean, 0.02, 0.2)

    # Weighted risk
    risk = 0.55*rms_score + 0.35*spec_score + 0.1*zcr_score
    return float(max(0, min(1, risk)))
