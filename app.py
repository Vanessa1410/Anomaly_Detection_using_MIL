from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory, jsonify
import os
import time
import threading
from werkzeug.utils import secure_filename
import tensorflow as tf
from model_utils import extract_frames, extract_features, classify_video, classify_anomaly_subtype

# Initialize Flask app
app = Flask(__name__)
app.secret_key = "supersecretkey"

# Configure directories
BASE_UPLOAD_FOLDER = "uploads"
VIDEO_FOLDER = os.path.join(BASE_UPLOAD_FOLDER, "videos")
FRAME_FOLDER = os.path.join(BASE_UPLOAD_FOLDER, "frames")
FEATURE_FOLDER = os.path.join(BASE_UPLOAD_FOLDER, "features")

# Store processing progress
processing_progress = {}

# Allowed file extensions
ALLOWED_EXTENSIONS = {'mp4', 'avi'}

# Ensure all directories exist
for folder in [BASE_UPLOAD_FOLDER, VIDEO_FOLDER, FRAME_FOLDER, FEATURE_FOLDER]:
    os.makedirs(folder, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    if "video" not in request.files:
        return jsonify({"status": "error", "message": "No file part"}), 400

    file = request.files["video"]
    if file.filename == "":
        return jsonify({"status": "error", "message": "No selected file"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(VIDEO_FOLDER, filename)
        file.save(file_path)
        
        # Mark the video as starting processing
        processing_progress[filename] = 10  # Start at 10%
        
        # Start processing in a new thread
        thread = threading.Thread(target=process_video, args=(filename,))
        thread.start()
        
        return jsonify({"status": "uploaded", "filename": filename}), 200

    return jsonify({"status": "error", "message": "Invalid file format"}), 400

def process_video(filename):
    from model_utils import load_models, mil_model, tcn_model  # ✅ Import inside function
    load_models()  # ✅ Ensure models are reloaded inside the thread

    try:
        video_path = os.path.join(VIDEO_FOLDER, filename)

        # Step 1: Extract Frames
        processing_progress[filename] = 30  
        frames_path = extract_frames(video_path, FRAME_FOLDER)
        time.sleep(2)

        # Step 2: Extract Features
        processing_progress[filename] = 60  
        features_path = extract_features(frames_path, FEATURE_FOLDER)
        time.sleep(2)

        # Step 3: Classification
        processing_progress[filename] = 90  

        print("🔍 Checking MIL Model:")
        if mil_model is None:
            raise ValueError("❌ MIL Model is still not loaded!")

        mil_model.summary()  # Ensure the model is loaded before use

        anomaly_detected, features_file = classify_video(features_path, video_path)

        if anomaly_detected:
            subtype_result = classify_anomaly_subtype(features_file)
            predicted_subtype = subtype_result["predicted_subtype"]
        else:
            predicted_subtype = None

        # Mark processing complete
        processing_progress[filename] = 100
        
    except Exception as e:
        processing_progress[filename] = -1  
        print(f"❌ Error processing video: {e}")


@app.route("/processing-status/<filename>")
def processing_status(filename):
    progress = processing_progress.get(filename, 0)
    if progress == 100:
        return jsonify({"status": "complete"})
    elif progress == -1:
        return jsonify({"status": "error"})
    return jsonify({"status": "processing", "progress": progress})

@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(VIDEO_FOLDER, filename)

@app.route("/results/<filename>")
def results(filename):
    video_path = os.path.join(VIDEO_FOLDER, filename)
    features_filename = f"{os.path.splitext(filename)[0]}_features.npy"
    features_path = os.path.join(FEATURE_FOLDER, features_filename)

    try:
        anomaly_detected, features_file = classify_video(features_path, video_path)
        if anomaly_detected:
            subtype_result = classify_anomaly_subtype(features_file)
            predicted_subtype = subtype_result["predicted_subtype"]
        else:
            predicted_subtype = None

    except Exception as e:
        flash(f"Error during classification: {e}")
        return redirect(url_for("index"))

    return render_template(
        "result.html",
        filename=filename,
        anomaly_detected=anomaly_detected,
        predicted_subtype=predicted_subtype,
    )

if __name__ == "__main__":
    app.run(debug=True)
