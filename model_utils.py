import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input

# Load your MIL model (adjust the path as necessary)
MIL_MODEL_PATH = 'models/mil_model_best_with_attention.h5'  # Replace with your actual MIL model path
TCN_MODEL_PATH = 'models/model_tcn_with_attention.h5'

# Load your MIL model
mil_model = None
tcn_model = None

def load_models():
    global mil_model, tcn_model
    
    if mil_model is None:  # Load MIL Model
        if os.path.exists(MIL_MODEL_PATH):
            mil_model = load_model(MIL_MODEL_PATH, compile=False)
            print("✅ MIL Model loaded successfully.")
        else:
            raise FileNotFoundError(f"❌ MIL model file not found at {MIL_MODEL_PATH}")

    if tcn_model is None:  # Load TCN Model
        if os.path.exists(TCN_MODEL_PATH):
            tcn_model = load_model(TCN_MODEL_PATH, compile=False)
            print("✅ TCN Model loaded successfully.")
        else:
            raise FileNotFoundError(f"❌ TCN model file not found at {TCN_MODEL_PATH}")

# ✅ Call `load_models()` immediately when this script is imported
load_models()


# EfficientNet model for feature extraction
def create_efficientnet_model():
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D()
    ])
    return model

efficientnet_model = create_efficientnet_model()

def extract_frames(video_path, frame_folder, target_frame_count=32):
    if not os.path.exists(frame_folder):
        os.makedirs(frame_folder)

    video_name = os.path.basename(video_path).split(".")[0]
    output_path = os.path.join(frame_folder, video_name)
    os.makedirs(output_path, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames - 1, target_frame_count, dtype=int)

    saved_frame_count = 0
    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        if i in frame_indices:
            frame_filename = os.path.join(output_path, f"frame_{saved_frame_count:04d}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_frame_count += 1

    cap.release()
    return output_path

def extract_features(frames_path, feature_folder, target_frame_count=32):
    if not os.path.exists(feature_folder):
        os.makedirs(feature_folder)

    features = []
    frame_files = sorted(os.listdir(frames_path))

    if len(frame_files) != target_frame_count:
        indices = np.linspace(0, len(frame_files) - 1, target_frame_count, dtype=int)
        frame_files = [frame_files[i] for i in indices]

    for frame_file in frame_files:
        frame_path = os.path.join(frames_path, frame_file)
        img = image.load_img(frame_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        
        feature = efficientnet_model.predict(img_array)
        flattened_feature = feature.flatten()
        features.append(flattened_feature)

    features = np.array(features)

    if features.shape[0] < target_frame_count:
        padding = np.zeros((target_frame_count - features.shape[0], 1280), dtype=features.dtype)
        features = np.vstack((features, padding))
    elif features.shape[0] > target_frame_count:
        features = features[:target_frame_count, :]

    feature_filename = f"{os.path.basename(frames_path)}_features.npy"
    feature_file = os.path.join(feature_folder, feature_filename)
    np.save(feature_file, features)
    return feature_file

def classify_video(features_path, video_path):
    global mil_model
    print(f"Loading features from {features_path}")
    features = np.load(features_path, allow_pickle=True)

    if features.shape != (32, 1280):
        if features.shape[0] < 32:
            padding = np.zeros((32 - features.shape[0], 1280), dtype=features.dtype)
            features = np.vstack((features, padding))
        else:
            features = features[:32, :]

    features = np.expand_dims(features, axis=0)
    print(f"Feature shape after batch dimension added: {features.shape}")

    prediction = mil_model.predict(features)
    print(f"Model prediction: {prediction}")

    threshold = 0.3  # Adjust threshold to detect more anomalies
    anomaly_detected = prediction[0] > threshold

    print(f"Anomaly detected (Threshold: {threshold}): {anomaly_detected}")
    return anomaly_detected, features_path

def classify_anomaly_subtype(features_path):
    global tcn_model
    features = np.load(features_path)

    if features.shape != (32, 1280):
        if features.shape[0] < 32:
            padding = np.zeros((32 - features.shape[0], 1280), dtype=features.dtype)
            features = np.vstack((features, padding))
        else:
            features = features[:32, :]

    features = np.expand_dims(features, axis=0)
    predictions = tcn_model.predict(features)
    predicted_class = np.argmax(predictions)
    confidence_scores = predictions[0]

    class_names = ["Fight", "Assault", "Collapse"]
    predicted_subtype = class_names[predicted_class]

    return {
        "predicted_subtype": predicted_subtype,
        "confidence_scores": {class_names[i]: float(confidence_scores[i]) for i in range(len(class_names))}
    }
