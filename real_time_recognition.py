import cv2
from deepface import DeepFace

# Path to your database of known faces
db_path = "./database"

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    try:
        # The find function will search for faces in the frame within the database
        dfs = DeepFace.find(img_path=frame, db_path=db_path, enforce_detection=False, silent=True)

        if dfs and not dfs[0].empty:
            for _, row in dfs[0].iterrows():
                if 'identity' in row and 'source_x' in row:
                    identity = row['identity']
                    x, y, w, h = row['source_x'], row['source_y'], row['source_w'], row['source_h']

                    # Draw a rectangle around the face
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    # Put the name of the person
                    # Extracting the name from the file path
                    name = identity.split('/')[-1].split('.')[0]
                    cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    except Exception as e:
        # Using a silent try-except block to handle frames with no faces gracefully
        pass

    cv2.imshow('Real-time Face Recognition', frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
