# Real-Time Face Verification System

This project is a real-time face verification system built with a Python Flask backend and a simple HTML/JavaScript frontend. It allows users to register their face and then verify their identity through a webcam.

## Project Structure

```
.
├── app.py              # The main Flask application file.
└── routes/
    ├── index.html      # Frontend for face verification.
    └── register.html   # Frontend for user registration.
```

## Backend (`app.py`)

The backend is a Flask application that handles face registration and verification using the `DeepFace` library.

### Endpoints

#### `POST /register`

Registers a new user in the system.

*   **Request Body:**
    ```json
    {
      "name": "string",
      "image": "string (base64 encoded)"
    }
    ```
*   **Responses:**
    *   `201 Created`: If the user is registered successfully.
    *   `400 Bad Request`: If the name is invalid, a user with that name already exists, or the request is missing data.
    *   `500 Internal Server Error`: For any other server-side errors.

#### `POST /verify`

Verifies a face from an incoming video stream against the registered users in the database.

*   **Request Body:**
    ```json
    {
      "image": "string (base64 encoded)"
    }
    ```
*   **Responses:**
    *   `200 OK`: Returns the verification status.
        *   `{"status": "Verified", "id": "user_name"}`
        *   `{"status": "Unverified", "message": "Unknown person."}`
        *   `{"status": "Failed", "message": "Spoof attempt detected."}`
        *   `{"status": "Error", "message": "..."}`
    *   `400 Bad Request`: If the request is missing data.
    *   `500 Internal Server Error`: For any other server-side errors.

### How it Works

1.  **Initialization:** On startup, the application pre-loads the necessary `DeepFace` models for faster processing and checks for the existence of the face database directory (`./database`).
2.  **Registration:** When a user registers, their name and image are sent to the `/register` endpoint. The application validates the input, detects the face in the image, and saves it in a new directory under `./database/{user_name}/face.jpg`.
3.  **Verification:** The frontend continuously captures frames from the webcam and sends them to the `/verify` endpoint. The backend performs face detection, anti-spoofing checks, and then uses `DeepFace.find()` to compare the detected face against the database of registered users.

## Frontend (`routes/`)

The frontend consists of two simple HTML pages with embedded JavaScript and CSS.

### `routes/index.html`

This is the main page for real-time face verification.

*   **Functionality:**
    *   Accesses the user's webcam.
    *   Continuously captures frames and sends them to the `/verify` endpoint.
    *   Displays the current verification status (`Initializing...`, `Verifying...`, `Verified`, `Unverified`, etc.) in a status box.

### `routes/register.html`

This page allows new users to register.

*   **Functionality:**
    *   Allows the user to enter their name.
    *   Provides two options for providing an image:
        1.  Capture a photo using the webcam.
        2.  Upload an image file (`.jpg`, `.png`).
    *   Sends the name and the captured/uploaded image to the `/register` endpoint.
    *   Displays the result of the registration attempt.

## How to Run

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/faraayeesha286941/Face.git
    ```

2.  **Navigate to the Project Directory:**
    ```bash
    cd Face
    ```

3.  **Set up a Virtual Environment:**

    It is highly recommended to run this project in a virtual environment to manage dependencies cleanly.

    *   **Create the virtual environment:**
        ```bash
        python -m venv venv
        ```

    *   **Activate the virtual environment:**
        *   On Windows:
            ```bash
            .\venv\Scripts\activate
            ```
        *   On macOS/Linux:
            ```bash
            source venv/bin/activate
            ```

4.  **Install Dependencies:**

    With your virtual environment activated, install the required packages from the `requirements.txt` file.
    ```bash
    pip install -r requirements.txt
    ```

5.  **Run the Backend Server:**
    ```bash
    python app.py
    ```
    The server will start on `http://127.0.0.1:5000`.

6.  **Open the Frontend:**
    Open the [`routes/register.html`](routes/register.html:1) file in your web browser to register a new user. Then, open [`routes/index.html`](routes/index.html:1) to start the verification process.