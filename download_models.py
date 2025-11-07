# download_models.py
from deepface import DeepFace

print("--- Starting Model Download and Build Process ---")
print("This may take several minutes depending on your internet connection.")
print("The script will download models for face detection, recognition, and anti-spoofing.")

try:
    # 1. Build the VGG-Face model for recognition
    print("\n[1/3] Building recognition model (VGG-Face)...")
    DeepFace.build_model("VGG-Face")
    print("VGG-Face model built successfully.")

    # 2. Build a backend model for detection
    print("\n[2/3] Building detector model (mtcnn)...")
    DeepFace.build_model("mtcnn") # mtcnn is a robust detector
    print("Detector model built successfully.")

    # 3. Force the anti-spoofing model to build by running it on a dummy image
    print("\n[3/3] Building anti-spoofing model...")
    # This call will trigger the download and build of the anti-spoofing models
    dummy_image = "https://raw.githubusercontent.com/serengil/deepface/master/tests/dataset/img1.jpg"
    DeepFace.extract_faces(img_path=dummy_image, anti_spoofing=True)
    print("Anti-spoofing model built successfully.")

    print("\n--- ALL MODELS ARE BUILT AND READY. You can now run the main script. ---")

except Exception as e:
    print(f"\n---!!! AN ERROR OCCURRED DURING MODEL SETUP !!!---")
    print(f"ERROR: {e}")
    print("Please check your internet connection and try again.")
    print("If the problem persists, the error message above is the key to solving it.")
