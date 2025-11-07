import cv2
import os
import glob
import numpy as np
import base64
import re
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from deepface import DeepFace

# --- Configuration ---
# The path to your face database.
DB_PATH = "./database"

# --- Flask App Initialization ---
# --- Flask App Initialization ---
app = Flask(__name__)

# --- CORS Configuration ---
# This is a more robust way to handle CORS and is recommended.
# It explicitly allows requests from any origin to any route on your API.
# For production, you might want to restrict this to your frontend's domain.
# For example: origins=["http://127.0.0.1:5500", "http://localhost:5500"]
CORS(app, resources={r"/*": {"origins": "*"}})

# --- Pre-load and Configure Models ---
# This dictionary will hold our pre-loaded models to ensure they are
# initialized only once when the application starts.
models = {}

def resize_image(image, max_size=1024):
    """
    Resizes an image to a maximum size, preserving aspect ratio.
    """
    h, w = image.shape[:2]
    if max(h, w) > max_size:
        ratio = max_size / max(h, w)
        new_w = int(w * ratio)
        new_h = int(h * ratio)
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return image

def initialize_backend():
    """
    Initializes the backend by checking the database and pre-loading necessary models.
    This function runs once at startup.
    """
    print("-> Initializing backend...")

    # --- Path and Verification ---
    db_path_abs = os.path.abspath(DB_PATH)
    if not os.path.isdir(db_path_abs):
        print(f"-> Database directory '{db_path_abs}' not found. Creating it.")
        os.makedirs(db_path_abs)

    # --- Database Update Check ---
    # This automatically forces a rebuild if new images are added.
    print("-> Checking for database updates...")
    try:
        # Check only if there are subdirectories (users) in the database
        if any(os.path.isdir(os.path.join(db_path_abs, i)) for i in os.listdir(db_path_abs)):
            pkl_files = glob.glob(os.path.join(db_path_abs, "representations_*.pkl"))
            if pkl_files:
                pkl_file_path = pkl_files[0]
                pkl_mod_time = os.path.getmtime(pkl_file_path)

                image_extensions = ["*.jpg", "*.jpeg", "*.png"]
                all_image_files = []
                for ext in image_extensions:
                    all_image_files.extend(glob.glob(os.path.join(db_path_abs, "**", ext), recursive=True))

                if all_image_files: # Proceed only if images are found
                    rebuild_needed = any(os.path.getmtime(img_path) > pkl_mod_time for img_path in all_image_files)

                    if rebuild_needed:
                        print(f"-> Detected database changes. Deleting '{os.path.basename(pkl_file_path)}' to force rebuild.")
                        os.remove(pkl_file_path)
            else:
                print("-> No representations file (.pkl) found. A new one will be created on the first run.")
        else:
            print("-> Database is empty. Skippingpkl check.")

    except Exception as e:
        print(f"---!!! WARNING: Could not check for database updates: {e} !!!---")

    # --- Pre-load DeepFace models for faster processing ---
    print("-> Pre-loading AI models. This may take a moment...")
    try:
        # Use a dummy find operation to pre-load and cache the models
        if any(os.path.isdir(os.path.join(db_path_abs, i)) for i in os.listdir(db_path_abs)):
             DeepFace.find(
                img_path=np.zeros([100, 100, 3], dtype=np.uint8),
                db_path=DB_PATH,
                enforce_detection=False,
                silent=True
            )
        else:
            print("-> Database empty. Skipping model pre-loading to avoid errors.")
        print("-> Models loaded successfully.")
    except Exception as e:
        print(f"---!!! WARNING: Could not pre-load models: {e} !!!---")


@app.route('/register', methods=['POST'])
def register_user():
    """
    API endpoint to register a new user with their name and face.
    Expects a JSON payload with 'name' and 'image' keys.
    """
    if not request.json or 'image' not in request.json or 'name' not in request.json:
        return jsonify({"error": "Bad Request: Missing 'image' or 'name' in JSON payload."}), 400

    name = request.json['name']
    image_b64 = request.json['image']

    # --- Input validation ---
    if not name or not re.match("^[a-zA-Z0-9_-]+$", name):
        return jsonify({"status": "Error", "message": "Invalid name. Use only letters, numbers, underscores, or hyphens."}), 400

    # --- Create user directory ---
    user_dir = os.path.join(DB_PATH, name)
    if os.path.exists(user_dir):
        return jsonify({"status": "Error", "message": f"User '{name}' already exists."}), 400

    try:
        # --- Decode image and save ---
        image_data = base64.b64decode(image_b64)
        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        frame = resize_image(frame) # Resize before processing

        # --- Validate that there is a detectable face ---
        face_objs = DeepFace.extract_faces(
            img_path=frame,
            detector_backend="mtcnn",
            enforce_detection=False # Set to False to check confidence manually
        )

        # Ensure a high-quality face is detected for registration
        if not face_objs or face_objs[0]['confidence'] < 0.95:
             return jsonify({"status": "Error", "message": "No clear face detected. Please provide a better image."}), 200

        # --- Create directory and save image ---
        os.makedirs(user_dir)
        output_path = os.path.join(user_dir, "face.jpg")
        cv2.imwrite(output_path, frame)

        print(f"-> User '{name}' registered successfully. Image saved to '{output_path}'.")
        # Invalidate the pkl file to force a rebuild on the next verification
        pkl_files = glob.glob(os.path.join(DB_PATH, "representations_*.pkl"))
        if pkl_files:
            os.remove(pkl_files[0])
            print("-> Removed representations.pkl to force a rebuild.")

        return jsonify({"status": "Success", "message": f"User {name} registered successfully!"}), 201

    except Exception as e:
        print(f"---!!! ERROR during registration: {e} !!!---")
        # Clean up created directory on failure
        if os.path.exists(user_dir):
            os.rmdir(user_dir)
        return jsonify({"error": f"An internal server error occurred: {e}"}), 500


@app.route('/verify', methods=['POST'])
def verify_face():
    """
    API endpoint to verify a face from an image.
    Expects a JSON payload with an 'image' key containing a base64 encoded string.
    """
    if not request.json or 'image' not in request.json:
        return jsonify({"error": "Bad Request: Missing 'image' in JSON payload."}), 400

    # Check if the database is empty before proceeding
    if not any(os.path.isdir(os.path.join(DB_PATH, i)) for i in os.listdir(DB_PATH)):
        return jsonify({"status": "Error", "message": "Database is empty. Please register a user first."}), 200

    try:
        # --- Decode the image from base64 ---
        image_data = base64.b64decode(request.json['image'])
        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        frame = resize_image(frame) # Resize before processing

        # --- Face Analysis and Recognition ---
        face_objs = DeepFace.extract_faces(
            img_path=frame,
            detector_backend="mtcnn",
            anti_spoofing=True,
            enforce_detection=False
        )

        # A higher confidence threshold helps filter out false positives on noisy or empty frames.
        if not face_objs or face_objs[0]['confidence'] < 0.95:
            return jsonify({"status": "Unverified", "message": "No face detected."}), 200

        face_obj = face_objs[0]

        # --- Anti-Spoofing Check ---
        if not face_obj['is_real']:
            return jsonify({"status": "Failed", "message": "Spoof attempt detected."}), 200

        # --- Face Recognition ---
        facial_area = face_obj['facial_area']
        x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']
        face_roi = frame[y:y+h, x:x+w]

        dfs = DeepFace.find(
            img_path=face_roi,
            db_path=DB_PATH,
            enforce_detection=False,
            silent=True
        )

        if len(dfs) > 0 and not dfs[0].empty:
            identity = dfs[0]['identity'][0]
            name = os.path.basename(os.path.dirname(identity))
            return jsonify({"status": "Verified", "id": name}), 200
        else:
            return jsonify({"status": "Unverified", "message": "Unknown person."}), 200

    except Exception as e:
        print(f"---!!! ERROR during verification: {e} !!!---")
        return jsonify({"error": "An internal server error occurred."}), 500

if __name__ == '__main__':
    initialize_backend()
    app.run(host='0.0.0.0', port=5000)