import cv2
import os
import glob
from deepface import DeepFace

# --- Configuration ---
# The path to your face database.
db_path = "./database"

# --- Path and Verification ---
# Convert to an absolute path to avoid any issues.
db_path = os.path.abspath(db_path)

# This check is essential, so we'll keep the error message if it fails.
if not os.path.isdir(db_path) or not os.listdir(db_path):
    print("---!!! FATAL ERROR !!!---")
    print(f"The database path '{db_path}' is wrong, does not exist, or is empty.")
    print("Please create the folder and, inside it, create a sub-folder with a person's name and their picture.")
    exit()

# --- NEW: Pre-run Check for Database Updates ---
# This section checks for new or modified images and removes the .pkl file to force a rebuild.
print("-> Checking for database updates...")
try:
    # Find any existing representation pickle files.
    pkl_files = glob.glob(os.path.join(db_path, "representations_*.pkl"))

    if pkl_files:
        pkl_file_path = pkl_files[0]
        pkl_mod_time = os.path.getmtime(pkl_file_path)

        # Get a list of all image files in the database, recursively.
        image_extensions = ["*.jpg", "*.jpeg", "*.png"]
        all_image_files = []
        for ext in image_extensions:
            all_image_files.extend(glob.glob(os.path.join(db_path, "**", ext), recursive=True))

        # Check if any image file is newer than the .pkl file.
        rebuild_needed = False
        for img_path in all_image_files:
            if os.path.getmtime(img_path) > pkl_mod_time:
                print(f"-> Detected new or modified file: {os.path.basename(img_path)}")
                rebuild_needed = True
                break

        if rebuild_needed:
            print(f"-> Deleting '{os.path.basename(pkl_file_path)}' to force a database rebuild.")
            os.remove(pkl_file_path)
    else:
        print("-> No representations file (.pkl) found. A new one will be created automatically.")

except Exception as e:
    print(f"---!!! WARNING !!!---")
    print(f"-> An error occurred during the database check: {e}")
# --- END OF NEW SECTION ---


# --- Webcam Initialization ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("---!!! FATAL ERROR !!!---")
    print("Cannot open webcam. Check if it's connected or used by another program.")
    exit()

print("-> Stream starting. Press 'q' in the window to quit.")

# --- Main Loop ---
while True:
    ret, frame = cap.read()
    if not ret:
        break # Exit loop if the stream ends

    try:
        # Detect faces and run the anti-spoofing check
        face_objs = DeepFace.extract_faces(
            img_path=frame,
            detector_backend="mtcnn",
            anti_spoofing=True,
            enforce_detection=False
        )

        for face_obj in face_objs:
            facial_area = face_obj['facial_area']
            x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']

            if face_obj['is_real']:
                # If face is REAL (green box)
                color = (0, 255, 0)

                # Perform recognition
                # If the .pkl file was deleted, DeepFace.find will automatically recreate it here.
                dfs = DeepFace.find(
                    img_path=frame[y:y+h, x:x+w],
                    db_path=db_path,
                    enforce_detection=False,
                    silent=True
                )

                if dfs and not dfs[0].empty:
                    # Match found
                    identity = dfs[0]['identity'][0]
                    name = os.path.basename(os.path.dirname(identity))
                    label = name
                else:
                    # No match found
                    label = "Unknown"

            else:
                # If face is a SPOOF (red box)
                color = (0, 0, 255)
                label = "SPOOF"

            # Draw the box and the label
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    except Exception:
        # Silently ignore any errors in the loop to keep the stream running
        pass

    # Show the final frame
    cv2.imshow("Face Recognition", frame)

    # Quit if 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# --- Cleanup ---
cap.release()
cv2.destroyAllWindows()
print("Stream finished.")
